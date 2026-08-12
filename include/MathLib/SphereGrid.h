///----------------------------------------
///       @file SphereGrid.h
///    @ingroup MathLib
///      @brief A fixed longitude/latitude quadtree over the unit sphere, with frustum queries.
///    @details The grid divides the sphere into 4×2 cells at level 0 and doubles both dimensions
///             per level, up to level @ref SphereGrid::maxLevel (6), a 256×128 grid. Cell corner
///             directions come from one shared table sampled at the deepest level, so a corner
///             shared between cells — or between a parent and its children — is the *same* vector
///             at every level, bit for bit. That identity is what lets a caller stitch geometry
///             generated at different levels without cracks.
///
///             The frustum query, @ref SphereGrid::visitCells, walks the quadtree against a sphere
///             placed anywhere in space: a cell that faces away from the viewer or whose bounding
///             sphere misses the frustum prunes its whole subtree, and visible cells are refined
///             down to a caller-chosen leaf level. It reports leaf-resolution spans, so a caller
///             maintaining per-cell state can clear an entire pruned block in one range operation.
///
///             Longitude 0 points along +z and the sine of longitude is negated — the left-handed
///             convention the wobbleworks renderers use throughout. @ref cellForLonLat maps a
///             longitude/latitude pair to the cell that contains it under that same convention.
///       @note The corner table is 1.25 cycles of cosine tapped at four offsets to serve as all four
///             sine/cosine tables, with the quadrant entries forced exact so seams land on
///             exact axes. The self-test regenerates it from @c std::sin / @c std::cos and
///             verifies both the values and the exact quadrants.
///     @author Created by John Stephen (wobbleworks.com)
///  @copyright Copyright © 2026 John Stephen (wobbleworks.com)
///             Licensed under the Apache License, Version 2.0.
///             SPDX-License-Identifier: Apache-2.0
///----------------------------------------

#pragma once

#include "MathLib/Frustum.h"
#include "MathLib/Numbers.h"
#include "MathLib/SelfTestCheck.h"
#include "MathLib/Vector.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <concepts>
#include <numbers>

///----------------------------------------
namespace Math {
///----------------------------------------

///----------------------------------------
///   @class SphereGrid
///   @brief A stateless longitude/latitude quadtree over the unit sphere.
/// @details All members are static; the class is a namespace with access control. Level @c level
///          has @ref columnCount(level) × @ref rowCount(level) cells; column 0 starts at
///          longitude 0 (+z) and row 0 starts at the south pole (−y).
///----------------------------------------

class SphereGrid final {
///----------------------------------------
public:
	SphereGrid() = delete;
	
	/// @brief The deepest subdivision level the corner table supports: a 256×128 grid.
	static constexpr auto maxLevel = 6;
	
	///----------------------------------------
	/// @brief Number of longitude columns at @p level (4 at level 0, doubling per level).
	///----------------------------------------
	
	[[nodiscard]] static constexpr int columnCount(int level) noexcept {
		return 4 << level;
	}
	
	///----------------------------------------
	/// @brief Number of latitude rows at @p level (2 at level 0, doubling per level).
	///----------------------------------------
	
	[[nodiscard]] static constexpr int rowCount(int level) noexcept {
		return 2 << level;
	}
	
	///----------------------------------------
	///  @struct Cell
	///   @brief One quadtree cell: its grid coordinates and unit-sphere corner directions.
	/// @details Corners shared with a neighbouring cell, or with the cell's parent or children,
	///          are bit-identical vectors — every level samples the same underlying table.
	///          At a pole the two corners on the polar row collapse to (0, ±1, 0).
	///----------------------------------------
	
	struct Cell {
		int level = 0;
		int column = 0;
		int row = 0;
		Double3 bottomLeftDirection;
		Double3 bottomRightDirection;
		Double3 topLeftDirection;
		Double3 topRightDirection;
		
		///----------------------------------------
		///   @brief The mean of the four corner directions.
		/// @details Not normalized — it lies slightly inside the unit sphere, which is exactly
		///          what the visibility test wants: scaled by a sphere's radius it is the centre
		///          of the cell's surface patch.
		///----------------------------------------
		
		[[nodiscard]] constexpr Double3 centerDirection() const noexcept {
			return (bottomLeftDirection + bottomRightDirection + topLeftDirection + topRightDirection) * 0.25;
		}
		
		///----------------------------------------
		///      @brief The smallest corner-covering sphere of this cell on a given world sphere.
		///    @details Centred on the cell's surface-patch centre with a radius reaching the
		///             farthest corner. The corner directions already live on the unit sphere, so
		///             the work happens there and the result is scaled and offset onto the world
		///             sphere.
		///      @param sphereCenter Centre of the world sphere the grid is applied to.
		///      @param sphereRadius Radius of the world sphere.
		/// @param[out] center Centre of the bounding sphere.
		/// @param[out] radius Radius of the bounding sphere.
		///----------------------------------------
		
		void boundingSphere(const Double3 &sphereCenter, double sphereRadius, Double3 &center, double &radius) const noexcept {
			// Work on the unit sphere, then scale and offset onto the world sphere
			auto unitCenter = centerDirection();
			center = sphereCenter + unitCenter * sphereRadius;
			
			// The radius reaches whichever corner lies farthest from the patch centre
			auto bottomLeftOffset = bottomLeftDirection - unitCenter;
			auto bottomRightOffset = bottomRightDirection - unitCenter;
			auto topLeftOffset = topLeftDirection - unitCenter;
			auto topRightOffset = topRightDirection - unitCenter;
			auto bottomLeftDistSquared = dot(bottomLeftOffset, bottomLeftOffset);
			auto bottomRightDistSquared = dot(bottomRightOffset, bottomRightOffset);
			auto topLeftDistSquared = dot(topLeftOffset, topLeftOffset);
			auto topRightDistSquared = dot(topRightOffset, topRightOffset);
			
			auto maxDistanceSquared = std::max(std::max(bottomLeftDistSquared, bottomRightDistSquared), std::max(topLeftDistSquared, topRightDistSquared));
			radius = std::sqrt(maxDistanceSquared) * sphereRadius;
		}
	};
	
	///----------------------------------------
	///  @struct CellSpan
	///   @brief A rectangle of leaf-level cells sharing one visibility verdict.
	/// @details @ref visitCells reports visible cells as 1×1 spans and pruned subtrees as square
	///          power-of-two blocks, always in leaf-level coordinates. Across one traversal the
	///          spans partition the whole leaf grid: every leaf cell appears in exactly one span.
	///----------------------------------------
	
	struct CellSpan {
		int column = 0;       ///< First column of the span, in leaf-level coordinates.
		int row = 0;          ///< First row of the span, in leaf-level coordinates.
		int columnCount = 0;  ///< Number of columns covered.
		int rowCount = 0;     ///< Number of rows covered.
		bool visible = false; ///< Whether the covered cells may be visible.
	};
	
	///----------------------------------------
	/// @struct CellCoordinate
	///  @brief A cell position within a grid of a given size.
	///----------------------------------------
	
	struct CellCoordinate {
		int column = 0;
		int row = 0;
	};
	
	///----------------------------------------
	///   @brief The cell at @p column, @p row of level @p level.
	/// @details Corner directions are table lookups sampled at the deepest level, so corners
	///          shared between cells or across levels are bit-identical.
	///   @param column Column in [0, @ref columnCount(level)).
	///   @param row Row in [0, @ref rowCount(level)).
	///   @param level Subdivision level in [0, @ref maxLevel].
	///----------------------------------------
	
	[[nodiscard]] static constexpr Cell cell(int column, int row, int level) noexcept {
		assert(level >= 0 && level <= maxLevel);
		assert(column >= 0 && column < columnCount(level));
		assert(row >= 0 && row < rowCount(level));
		
		// Sample the deepest-level table at this level's stride
		auto indexStride = 1 << (maxLevel - level);
		auto rowIndex = row * indexStride;
		auto columnIndex = column * indexStride;
		
		return Cell{
			.level = level,
			.column = column,
			.row = row,
			.bottomLeftDirection = cornerDirection(columnIndex, rowIndex),
			.bottomRightDirection = cornerDirection(columnIndex + indexStride, rowIndex),
			.topLeftDirection = cornerDirection(columnIndex, rowIndex + indexStride),
			.topRightDirection = cornerDirection(columnIndex + indexStride, rowIndex + indexStride)
		};
	}
	
	///----------------------------------------
	///   @brief The cell containing a longitude/latitude position, for an arbitrary grid size.
	/// @details Longitude wraps into [0, 2π) and latitude clamps to ±π/2; positions landing on
	///          the far edge (longitude 2π, latitude +π/2) fold into the last cell.
	///     @note Deliberately computed in @c float: the shipped label databases were generated
	///           with this exact single-precision arithmetic, and reader and writer must agree
	///           on which side of a cell boundary a position falls.
	///   @param numColumns Number of longitude columns in the grid.
	///   @param numRows Number of latitude rows in the grid.
	///   @param longitude Longitude in radians; any real value.
	///   @param latitude Latitude in radians, −π/2 at the south pole.
	///----------------------------------------
	
	[[nodiscard]] static CellCoordinate cellForLonLat(int numColumns, int numRows, float longitude, float latitude) noexcept {
		assert(numColumns > 0);
		assert(numRows > 0);
		
		// Wrap the longitude into [0, 2π) and clamp the latitude to the poles
		constexpr auto twoPi = std::numbers::pi_v<float> * 2.0f;
		constexpr auto halfPi = std::numbers::pi_v<float> * 0.5f;
		auto wrappedLongitude = std::fmod(longitude, twoPi);
		if (wrappedLongitude < 0.0f) {
			wrappedLongitude += twoPi;
		}
		auto clampedLatitude = std::clamp(latitude, -halfPi, halfPi);
		
		// Scale into grid coordinates
		auto columnSpacing = twoPi / float(numColumns);
		auto rowSpacing = std::numbers::pi_v<float> / float(numRows);
		auto column = int(std::floor(wrappedLongitude / columnSpacing));
		auto row = int(std::floor((clampedLatitude + halfPi) / rowSpacing));
		
		// Fold the far edge into the last cell
		return CellCoordinate{
			.column = std::min(column, numColumns - 1),
			.row = std::min(row, numRows - 1)
		};
	}
	
	///----------------------------------------
	///   @brief Walks the quadtree against @p frustum, reporting leaf-level spans to @p visitor.
	/// @details The grid is applied to the sphere at @p sphereCenter of @p sphereRadius. A cell
	///          whose surface patch faces away from the frustum's apex, or whose bounding sphere
	///          misses the frustum, is pruned: its whole subtree is reported once as an invisible
	///          span. Visible cells are refined down to @p leafLevel and reported as visible 1×1
	///          spans. The spans partition the leaf grid — every leaf cell is reported exactly
	///          once — so a caller can maintain per-cell state with no gaps or double visits.
	///     @note The visibility test is conservative in the usual frustum-culling sense: a span
	///           reported visible may prove empty on screen, but a span reported invisible is
	///           guaranteed to be.
	///   @param sphereCenter Centre of the sphere the grid is applied to.
	///   @param sphereRadius Radius of the sphere.
	///   @param frustum The viewing volume, with its apex at the viewer.
	///   @param leafLevel The level to refine visible cells down to, in [0, @ref maxLevel].
	///   @param visitor Invoked once per span.
	///----------------------------------------
	
	template <typename Visitor> requires std::invocable<Visitor &, const CellSpan &>
	static void visitCells(const Double3 &sphereCenter, double sphereRadius, const Frustum64 &frustum, int leafLevel, Visitor &&visitor) {
		assert(leafLevel >= 0 && leafLevel <= maxLevel);
		
		// Walk each root cell of the 4×2 top-level grid
		for (int row = 0; row < rowCount(0); ++row) {
			for (int column = 0; column < columnCount(0); ++column) {
				visitCell(cell(column, row, 0), sphereCenter, sphereRadius, frustum, leafLevel, visitor);
			}
		}
	}
	
private:
	///----------------------------------------
	///   @brief Recursively classifies one cell, refining visible cells to the leaf level.
	///----------------------------------------
	
	template <typename Visitor>
	static void visitCell(const Cell &cell, const Double3 &sphereCenter, double sphereRadius, const Frustum64 &frustum, int leafLevel, Visitor &visitor) {
		// An invisible cell prunes its whole subtree as one leaf-level span
		if (!cellMayBeVisible(cell, sphereCenter, sphereRadius, frustum)) {
			auto stride = 1 << (leafLevel - cell.level);
			visitor(CellSpan{
				.column = cell.column * stride,
				.row = cell.row * stride,
				.columnCount = stride,
				.rowCount = stride,
				.visible = false
			});
			return;
		}
		
		// A visible leaf is reported individually
		if (cell.level == leafLevel) {
			visitor(CellSpan{
				.column = cell.column,
				.row = cell.row,
				.columnCount = 1,
				.rowCount = 1,
				.visible = true
			});
			return;
		}
		
		// A visible interior cell refines into its 2×2 children
		for (int subRow = 0; subRow < 2; ++subRow) {
			for (int subColumn = 0; subColumn < 2; ++subColumn) {
				visitCell(SphereGrid::cell(cell.column * 2 + subColumn, cell.row * 2 + subRow, cell.level + 1),
				          sphereCenter, sphereRadius, frustum, leafLevel, visitor);
			}
		}
	}
	
	///----------------------------------------
	///   @brief Whether a cell's surface patch could be visible from @p frustum.
	/// @details Two tests, both conservative. A patch whose every corner direction points the
	///          same way as the viewer-to-patch direction is on the far side of the sphere —
	///          using all four corners guarantees a useful verdict even where longitudes pinch
	///          together at a pole. What survives that is tested as its bounding sphere against
	///          the frustum.
	///----------------------------------------
	
	[[nodiscard]] static bool cellMayBeVisible(const Cell &cell, const Double3 &sphereCenter, double sphereRadius, const Frustum64 &frustum) {
		// The direction from the viewer to the cell's surface patch
		auto center = sphereCenter + cell.centerDirection() * sphereRadius;
		auto direction = center - frustum.apex();
		
		// If all four corners point the same way the patch faces away from the viewer
		if (dot(direction, cell.bottomLeftDirection) > 0.
		&& dot(direction, cell.bottomRightDirection) > 0.
		&& dot(direction, cell.topLeftDirection) > 0.
		&& dot(direction, cell.topRightDirection) > 0.) {
			return false;
		}
		
		// Otherwise the cell is visible exactly when its bounding sphere touches the frustum
		double radius;
		cell.boundingSphere(sphereCenter, sphereRadius, center, radius);
		return frustum.intersectsSphere(center, radius);
	}
	
	///----------------------------------------
	///   @brief The unit-sphere direction of a deepest-level grid corner.
	///   @param columnIndex Longitude index in [0, 256].
	///   @param rowIndex Latitude index in [0, 128].
	///----------------------------------------
	
	[[nodiscard]] static constexpr Double3 cornerDirection(int columnIndex, int rowIndex) noexcept {
		return Double3(sinLon(columnIndex) * cosLat(rowIndex), sinLat(rowIndex), cosLon(columnIndex) * cosLat(rowIndex));
	}
	
	///----------------------------------------
	/// @brief Table taps: cosine of longitude, negated sine of longitude, sine and cosine of
	///        latitude, all indexed at the deepest level. See the file note on the table layout.
	///----------------------------------------
	
	[[nodiscard]] static constexpr double cosLon(int index) noexcept { return sinCosTable[index]; }
	[[nodiscard]] static constexpr double sinLon(int index) noexcept { return sinCosTable[index + 64]; }
	[[nodiscard]] static constexpr double sinLat(int index) noexcept { return sinCosTable[index + 128]; }
	[[nodiscard]] static constexpr double cosLat(int index) noexcept { return sinCosTable[index + 192]; }
	
	///----------------------------------------
	///   @brief 1.25 cycles of cosine at the deepest level's spacing, tapped four ways.
	/// @details Because sine and cosine are phase-shifted copies of each other, one 321-entry
	///          table serves as cos(longitude) [0..256], −sin(longitude) [64..320],
	///          sin(latitude) [128..256] and cos(latitude) [192..320]. Quadrant entries are
	///          exact so cell seams land on exact axes.
	///----------------------------------------
	
	static constexpr std::array<double, 321> sinCosTable = {
		1.00000000000000000, 0.99969881869620425, 0.99879545620517241, 0.99729045667869021,
		0.99518472667219693, 0.99247953459870997, 0.98917650996478101, 0.98527764238894122,
		0.98078528040323043, 0.97570213003852857, 0.97003125319454397, 0.96377606579543984,
		0.95694033573220882, 0.94952818059303667, 0.94154406518302081, 0.93299279883473896,
		0.92387953251128674, 0.91420975570353069, 0.90398929312344334, 0.89322430119551532,
		0.88192126434835505, 0.87008699110871146, 0.85772861000027212, 0.84485356524970712,
		0.83146961230254524, 0.81758481315158371, 0.80320753148064494, 0.78834642762660634,
		0.77301045336273699, 0.75720884650648457, 0.74095112535495922, 0.72424708295146700,
		0.70710678118654757, 0.68954054473706694, 0.67155895484701844, 0.65317284295377687,
		0.63439328416364560, 0.61523159058062693, 0.59569930449243347, 0.57580819141784545,
		0.55557023301960229, 0.53499761988709738, 0.51410274419322177, 0.49289819222978415,
		0.47139673682599781, 0.44961132965460660, 0.42755509343028220, 0.40524131400498981,
		0.38268343236508984, 0.35989503653498828, 0.33688985339222005, 0.31368174039889152,
		0.29028467725446228, 0.26671275747489842, 0.24298017990326398, 0.21910124015686977,
		0.19509032201612830, 0.17096188876030136, 0.14673047445536175, 0.12241067519921628,
		0.09801714032956077, 0.07356456359966745, 0.04906767432741813, 0.02454122852291227,
		0.00000000000000000, -0.02454122852291214, -0.04906767432741801, -0.07356456359966734,
		-0.09801714032956066, -0.12241067519921617, -0.14673047445536164, -0.17096188876030124,
		-0.19509032201612819, -0.21910124015686966, -0.24298017990326387, -0.26671275747489831,
		-0.29028467725446222, -0.31368174039889146, -0.33688985339221994, -0.35989503653498817,
		-0.38268343236508973, -0.40524131400498975, -0.42755509343028186, -0.44961132965460671,
		-0.47139673682599770, -0.49289819222978393, -0.51410274419322155, -0.53499761988709693,
		-0.55557023301960196, -0.57580819141784534, -0.59569930449243325, -0.61523159058062671,
		-0.63439328416364527, -0.65317284295377653, -0.67155895484701844, -0.68954054473706683,
		-0.70710678118654746, -0.72424708295146689, -0.74095112535495888, -0.75720884650648468,
		-0.77301045336273699, -0.78834642762660623, -0.80320753148064483, -0.81758481315158360,
		-0.83146961230254535, -0.84485356524970712, -0.85772861000027212, -0.87008699110871135,
		-0.88192126434835494, -0.89322430119551521, -0.90398929312344334, -0.91420975570353069,
		-0.92387953251128674, -0.93299279883473885, -0.94154406518302070, -0.94952818059303667,
		-0.95694033573220882, -0.96377606579543984, -0.97003125319454397, -0.97570213003852846,
		-0.98078528040323043, -0.98527764238894122, -0.98917650996478101, -0.99247953459870997,
		-0.99518472667219682, -0.99729045667869021, -0.99879545620517241, -0.99969881869620425,
		-1.00000000000000000, -0.99969881869620425, -0.99879545620517241, -0.99729045667869021,
		-0.99518472667219693, -0.99247953459870997, -0.98917650996478101, -0.98527764238894133,
		-0.98078528040323043, -0.97570213003852857, -0.97003125319454397, -0.96377606579543995,
		-0.95694033573220894, -0.94952818059303679, -0.94154406518302081, -0.93299279883473896,
		-0.92387953251128674, -0.91420975570353069, -0.90398929312344345, -0.89322430119551532,
		-0.88192126434835505, -0.87008699110871146, -0.85772861000027212, -0.84485356524970723,
		-0.83146961230254546, -0.81758481315158371, -0.80320753148064494, -0.78834642762660634,
		-0.77301045336273710, -0.75720884650648479, -0.74095112535495899, -0.72424708295146700,
		-0.70710678118654768, -0.68954054473706716, -0.67155895484701866, -0.65317284295377709,
		-0.63439328416364593, -0.61523159058062737, -0.59569930449243325, -0.57580819141784523,
		-0.55557023301960218, -0.53499761988709726, -0.51410274419322188, -0.49289819222978426,
		-0.47139673682599786, -0.44961132965460693, -0.42755509343028247, -0.40524131400499036,
		-0.38268343236509034, -0.35989503653498794, -0.33688985339221994, -0.31368174039889146,
		-0.29028467725446239, -0.26671275747489853, -0.24298017990326412, -0.21910124015687010,
		-0.19509032201612866, -0.17096188876030172, -0.14673047445536233, -0.12241067519921596,
		-0.09801714032956045, -0.07356456359966736, -0.04906767432741803, -0.02454122852291239,
		0.00000000000000000, 0.02454122852291202, 0.04906767432741767, 0.07356456359966700,
		0.09801714032956009, 0.12241067519921560, 0.14673047445536194, 0.17096188876030133,
		0.19509032201612828, 0.21910124015686974, 0.24298017990326376, 0.26671275747489820,
		0.29028467725446211, 0.31368174039889113, 0.33688985339221961, 0.35989503653498767,
		0.38268343236509000, 0.40524131400499003, 0.42755509343028220, 0.44961132965460660,
		0.47139673682599759, 0.49289819222978387, 0.51410274419322144, 0.53499761988709693,
		0.55557023301960184, 0.57580819141784489, 0.59569930449243280, 0.61523159058062693,
		0.63439328416364560, 0.65317284295377676, 0.67155895484701833, 0.68954054473706683,
		0.70710678118654746, 0.72424708295146667, 0.74095112535495877, 0.75720884650648423,
		0.77301045336273666, 0.78834642762660589, 0.80320753148064505, 0.81758481315158371,
		0.83146961230254524, 0.84485356524970712, 0.85772861000027201, 0.87008699110871135,
		0.88192126434835494, 0.89322430119551510, 0.90398929312344312, 0.91420975570353047,
		0.92387953251128652, 0.93299279883473896, 0.94154406518302081, 0.94952818059303667,
		0.95694033573220882, 0.96377606579543984, 0.97003125319454397, 0.97570213003852846,
		0.98078528040323032, 0.98527764238894111, 0.98917650996478090, 0.99247953459870997,
		0.99518472667219693, 0.99729045667869021, 0.99879545620517241, 0.99969881869620425,
		1.00000000000000000, 0.99969881869620425, 0.99879545620517241, 0.99729045667869021,
		0.99518472667219693, 0.99247953459871008, 0.98917650996478090, 0.98527764238894122,
		0.98078528040323043, 0.97570213003852857, 0.97003125319454397, 0.96377606579543995,
		0.95694033573220894, 0.94952818059303679, 0.94154406518302092, 0.93299279883473907,
		0.92387953251128663, 0.91420975570353058, 0.90398929312344334, 0.89322430119551532,
		0.88192126434835505, 0.87008699110871146, 0.85772861000027223, 0.84485356524970734,
		0.83146961230254557, 0.81758481315158404, 0.80320753148064528, 0.78834642762660612,
		0.77301045336273688, 0.75720884650648457, 0.74095112535495922, 0.72424708295146711,
		0.70710678118654768, 0.68954054473706716, 0.67155895484701866, 0.65317284295377709,
		0.63439328416364593, 0.61523159058062737, 0.59569930449243325, 0.57580819141784523,
		0.55557023301960218, 0.53499761988709726, 0.51410274419322188, 0.49289819222978426,
		0.47139673682599792, 0.44961132965460698, 0.42755509343028253, 0.40524131400499042,
		0.38268343236509039, 0.35989503653498800, 0.33688985339222000, 0.31368174039889152,
		0.29028467725446244, 0.26671275747489859, 0.24298017990326418, 0.21910124015687016,
		0.19509032201612872, 0.17096188876030177, 0.14673047445536239, 0.12241067519921603,
		0.09801714032956052, 0.07356456359966743, 0.04906767432741809, 0.02454122852291245,
		0.00000000000000000,
	};
};

///----------------------------------------
///   @brief Validates the corner table, corner sharing, coordinate mapping and frustum traversal.
///----------------------------------------

inline void sphereGridSelfTest() {
	using selftest::check;
	
	// Every corner of the deepest-level grid matches a fresh std::sin/std::cos regeneration.
	// (The generator forced the quadrant entries — where std::sin(pi) is ~1.2e-16, not 0 — to
	// exact zeros and ones, and the comparison multiplies two table entries, hence a tolerance
	// a little above one ulp.)
	{
		constexpr auto tableTolerance = 1.e-14;
		auto deepColumns = SphereGrid::columnCount(SphereGrid::maxLevel);
		auto deepRows = SphereGrid::rowCount(SphereGrid::maxLevel);
		auto cornersMatch = true;
		for (int row = 0; row < deepRows; ++row) {
			for (int column = 0; column < deepColumns; ++column) {
				// The bottom-left corner sweeps every table index; the far edges are covered
				// through each last cell's bottom-right and top-left corners below
				auto corner = SphereGrid::cell(column, row, SphereGrid::maxLevel).bottomLeftDirection;
				auto longitudeAngle = math::numbers::two_pi * double(column) / double(deepColumns);
				auto latitudeAngle = std::numbers::pi * double(row) / double(deepRows) - math::numbers::half_pi;
				auto expected = Double3(-std::sin(longitudeAngle) * std::cos(latitudeAngle),
				                        std::sin(latitudeAngle),
				                        std::cos(longitudeAngle) * std::cos(latitudeAngle));
				cornersMatch = cornersMatch
					&& std::fabs(corner.x - expected.x) <= tableTolerance
					&& std::fabs(corner.y - expected.y) <= tableTolerance
					&& std::fabs(corner.z - expected.z) <= tableTolerance;
			}
		}
		check(cornersMatch, "every deepest-level corner matches its regenerated direction");
		
		// The far edges wrap and cap exactly: longitude 2pi equals longitude 0, latitude +pi/2
		// is the north pole
		auto lastColumn = SphereGrid::cell(deepColumns - 1, 0, SphereGrid::maxLevel);
		check(lastColumn.bottomRightDirection == SphereGrid::cell(0, 0, SphereGrid::maxLevel).bottomLeftDirection, "the longitude seam wraps to bit-identical vectors");
		check(SphereGrid::cell(0, deepRows - 1, SphereGrid::maxLevel).topLeftDirection == Double3(0, 1, 0), "the top row caps at the exact north pole");
	}
	
	// Corner directions are unit length away from the poles, and exactly (0, ±1, 0) at them
	{
		auto lengthTolerance = 1.e-14;
		for (int level : {0, 3, SphereGrid::maxLevel}) {
			for (int row = 0; row < SphereGrid::rowCount(level); row += std::max(1, SphereGrid::rowCount(level) / 4)) {
				for (int column = 0; column < SphereGrid::columnCount(level); column += std::max(1, SphereGrid::columnCount(level) / 8)) {
					auto cell = SphereGrid::cell(column, row, level);
					check(std::fabs(length(cell.topLeftDirection) - 1.0) <= lengthTolerance, "corner directions are unit length");
				}
			}
		}
		check(SphereGrid::cell(0, 0, 0).bottomLeftDirection == Double3(0, -1, 0), "the south pole corner is exact");
		check(SphereGrid::cell(0, 1, 0).topLeftDirection == Double3(0, 1, 0), "the north pole corner is exact");
	}
	
	// Corners are bit-identical across neighbours and across levels
	{
		auto cell = SphereGrid::cell(2, 1, 3);
		check(cell.bottomRightDirection == SphereGrid::cell(3, 1, 3).bottomLeftDirection, "neighbouring cells share corner vectors exactly");
		check(cell.topLeftDirection == SphereGrid::cell(2, 2, 3).bottomLeftDirection, "vertically neighbouring cells do too");
		check(cell.bottomLeftDirection == SphereGrid::cell(4, 2, 4).bottomLeftDirection, "a cell and its first child share their anchor corner exactly");
		check(cell.topRightDirection == SphereGrid::cell(5, 3, 4).topRightDirection, "and the far corner matches the last child");
	}
	
	// cellForLonLat wraps, clamps, and folds the far edges into the last cell
	{
		constexpr auto halfPiFloat = std::numbers::pi_v<float> * 0.5f;
		auto atOrigin = SphereGrid::cellForLonLat(8, 4, 0.0f, 0.0f);
		check(atOrigin.column == 0 && atOrigin.row == 2, "longitude 0, latitude 0 lands at the equator's north side of column 0");
		auto wrapped = SphereGrid::cellForLonLat(8, 4, -0.01f, 0.0f);
		check(wrapped.column == 7, "a slightly negative longitude wraps into the last column");
		auto northPole = SphereGrid::cellForLonLat(8, 4, 0.0f, halfPiFloat);
		check(northPole.row == 3, "latitude +pi/2 folds into the last row");
		auto southPole = SphereGrid::cellForLonLat(8, 4, 0.0f, -halfPiFloat);
		check(southPole.row == 0, "latitude -pi/2 lands in the first row");
	}
	
	// The coordinate mapping and the cell geometry agree: a cell's centre direction maps back
	// to that cell's coordinates. This is the invariant label lookup depends on.
	{
		for (int level : {0, 2, 4}) {
			auto numColumns = SphereGrid::columnCount(level);
			auto numRows = SphereGrid::rowCount(level);
			for (int row = 0; row < numRows; row += std::max(1, numRows / 4)) {
				for (int column = 0; column < numColumns; column += std::max(1, numColumns / 8)) {
					auto direction = SphereGrid::cell(column, row, level).centerDirection();
					// Longitude under the left-handed convention: 0 at +z, sine negated
					auto longitude = std::atan2(-direction.x, direction.z);
					if (longitude < 0) {
						longitude += math::numbers::two_pi;
					}
					auto latitude = std::atan2(direction.y, std::hypot(direction.x, direction.z));
					auto mapped = SphereGrid::cellForLonLat(numColumns, numRows, float(longitude), float(latitude));
					check(mapped.column == column && mapped.row == row, "a cell centre maps back to its own coordinates");
				}
			}
		}
	}
	
	// Traversal: viewing a unit sphere from outside, the spans partition the leaf grid exactly,
	// the facing cell is visible, and the far side is culled
	{
		auto leafLevel = 3;
		auto numColumns = SphereGrid::columnCount(leafLevel);
		auto numRows = SphereGrid::rowCount(leafLevel);
		auto apex = Double3(0, 0, -3);
		auto viewer = Frustum64::perspective(apex, Double4x4::identity(), math::numbers::half_pi, math::numbers::half_pi);
		
		std::array<int, 32 * 16> coverage{};
		auto visibleCount = 0;
		SphereGrid::visitCells(Double3(0, 0, 0), 1.0, viewer, leafLevel, [&](const SphereGrid::CellSpan &span) {
			for (int row = span.row; row < span.row + span.rowCount; ++row) {
				for (int column = span.column; column < span.column + span.columnCount; ++column) {
					++coverage[row * numColumns + column];
				}
			}
			if (span.visible) {
				check(span.columnCount == 1 && span.rowCount == 1, "visible cells are reported individually");
				++visibleCount;
			}
		});
		auto everyCellOnce = true;
		for (auto count : coverage) {
			everyCellOnce = everyCellOnce && count == 1;
		}
		check(everyCellOnce, "the spans partition the leaf grid exactly");
		check(visibleCount > 0, "something is visible");
		check(visibleCount < numColumns * numRows / 2, "the far side and off-screen cells are culled");
		
		// The cell facing the viewer (longitude pi puts +z at the far side, so -z faces the
		// apex) is visible; the cell on the far side is not
		auto facing = SphereGrid::cellForLonLat(numColumns, numRows, std::numbers::pi_v<float>, 0.0f);
		auto farSide = SphereGrid::cellForLonLat(numColumns, numRows, 0.0f, 0.0f);
		auto facingVisible = false;
		auto farSideVisible = false;
		SphereGrid::visitCells(Double3(0, 0, 0), 1.0, viewer, leafLevel, [&](const SphereGrid::CellSpan &span) {
			auto contains = [&](const SphereGrid::CellCoordinate &coordinate) {
				return coordinate.column >= span.column && coordinate.column < span.column + span.columnCount
				&& coordinate.row >= span.row && coordinate.row < span.row + span.rowCount;
			};
			if (contains(facing) && span.visible) {
				facingVisible = true;
			}
			if (contains(farSide) && span.visible) {
				farSideVisible = true;
			}
		});
		check(facingVisible, "the cell facing the viewer is visible");
		check(!farSideVisible, "the cell on the far side of the sphere is culled");
	}
	
	// A leaf level of 0 reports exactly the eight root cells
	{
		auto apex = Double3(0, 0, -3);
		auto viewer = Frustum64::perspective(apex, Double4x4::identity(), math::numbers::half_pi, math::numbers::half_pi);
		auto spanCount = 0;
		SphereGrid::visitCells(Double3(0, 0, 0), 1.0, viewer, 0, [&](const SphereGrid::CellSpan &span) {
			check(span.columnCount == 1 && span.rowCount == 1, "level-0 spans are single cells");
			++spanCount;
		});
		check(spanCount == 8, "a leaf level of 0 reports the eight root cells");
	}
}

///----------------------------------------
} // namespace Math
///----------------------------------------
