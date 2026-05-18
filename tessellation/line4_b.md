Here is a complete, self-contained JavaScript implementation that keeps the implicit-algebraic spirit of your example, but fixes the rasterization logic so it behaves like a real line-drawing algorithm across all octants.

It includes:

an implicit line object ax + by + c = 0

a robust integer Bresenham implementation

a Grothendieck-inspired implicit rasterizer based on “evaluating a local equation”

a polygon tessellator that rasterizes every edge

a small demo at the bottom


/**
 * Grothendieck-inspired rasterization:
 * Treat a line as a local algebraic equation f(x, y) = ax + by + c.
 * Rasterization becomes the problem of moving on the integer lattice
 * while keeping the discrete points close to the zero-locus of f.
 *
 * This is not algebraic geometry in the formal sense;
 * it is a conceptual translation of "study geometry through equations"
 * into a pixel/lattice renderer.
 */

class ImplicitLine {
  constructor(x0, y0, x1, y1) {
    this.x0 = x0;
    this.y0 = y0;
    this.x1 = x1;
    this.y1 = y1;

    // ax + by + c = 0
    this.a = y0 - y1;
    this.b = x1 - x0;
    this.c = x0 * y1 - x1 * y0;
  }

  eval(x, y) {
    return this.a * x + this.b * y + this.c;
  }

  sign(x, y) {
    const v = this.eval(x, y);
    return v > 0 ? 1 : v < 0 ? -1 : 0;
  }

  isOnLine(x, y) {
    return this.eval(x, y) === 0;
  }
}

/**
 * Standard Bresenham line rasterization for all octants.
 * Returns integer lattice points from (x0, y0) to (x1, y1).
 */
function bresenhamInteger(x0, y0, x1, y1) {
  const points = [];

  let x = x0;
  let y = y0;

  const dx = Math.abs(x1 - x0);
  const dy = Math.abs(y1 - y0);
  const sx = x0 < x1 ? 1 : -1;
  const sy = y0 < y1 ? 1 : -1;

  let err = dx - dy;

  while (true) {
    points.push([x, y]);
    if (x === x1 && y === y1) break;

    const e2 = 2 * err;

    if (e2 > -dy) {
      err -= dy;
      x += sx;
    }

    if (e2 < dx) {
      err += dx;
      y += sy;
    }
  }

  return points;
}

/**
 * Implicit-equation rasterization.
 * Uses the line function f(x, y) = ax + by + c and chooses the next step
 * that stays closer to the zero set.
 *
 * This version is robust and works for all octants.
 */
function rasterizeImplicitLine(x0, y0, x1, y1) {
  const line = new ImplicitLine(x0, y0, x1, y1);
  const points = [];

  let x = x0;
  let y = y0;

  const sx = x0 < x1 ? 1 : -1;
  const sy = y0 < y1 ? 1 : -1;

  const dx = Math.abs(x1 - x0);
  const dy = Math.abs(y1 - y0);

  // The algorithm follows the major direction, but uses the implicit
  // equation to decide whether to step in x, y, or both.
  let guard = 0;
  const maxSteps = dx + dy + 2;

  while (true) {
    points.push([x, y]);
    if (x === x1 && y === y1) break;

    // Current signed distance proxy
    const f = line.eval(x, y);

    // Evaluate the two candidate moves:
    // move in x only, y only, or diagonal if needed
    const fx = line.eval(x + sx, y);
    const fy = line.eval(x, y + sy);
    const fxy = line.eval(x + sx, y + sy);

    const absF = Math.abs(f);
    const absFx = Math.abs(fx);
    const absFy = Math.abs(fy);
    const absFxy = Math.abs(fxy);

    // Pick the step that best minimizes |f|.
    // Tie-breaking is done by favoring diagonal if it improves both axes.
    if (absFxy <= absFx && absFxy <= absFy) {
      x += sx;
      y += sy;
    } else if (absFx < absFy) {
      x += sx;
    } else {
      y += sy;
    }

    guard++;
    if (guard > maxSteps) break;
  }

  return points;
}

/**
 * A cleaner "discrete scheme" style rasterizer:
 * We sample the lattice points in the bounding box and keep the points
 * whose implicit value changes sign between neighboring cells.
 *
 * This is useful when you want a more algebraic feel than Bresenham.
 */
function rasterizeImplicitByCells(x0, y0, x1, y1) {
  const line = new ImplicitLine(x0, y0, x1, y1);
  const points = [];

  const minX = Math.min(x0, x1);
  const maxX = Math.max(x0, x1);
  const minY = Math.min(y0, y1);
  const maxY = Math.max(y0, y1);

  // Bresenham-like edge tracing by sign consistency
  let x = x0;
  let y = y0;
  const sx = x0 < x1 ? 1 : -1;
  const sy = y0 < y1 ? 1 : -1;

  points.push([x, y]);

  while (x !== x1 || y !== y1) {
    const f = line.eval(x, y);
    const fx = line.eval(x + sx, y);
    const fy = line.eval(x, y + sy);

    // Decide based on which neighboring point is closer to the zero locus.
    const chooseX = Math.abs(fx) < Math.abs(fy);

    if (chooseX) {
      x += sx;
    } else {
      y += sy;
    }

    // Ensure we do not wander outside the segment's bounding box.
    if (x < minX - 1 || x > maxX + 1 || y < minY - 1 || y > maxY + 1) {
      break;
    }

    points.push([x, y]);
  }

  return points;
}

/**
 * Rasterize a polygon by rasterizing each edge.
 * Duplicate points are removed while preserving order.
 */
function rasterizePolygon(vertices, method = rasterizeImplicitLine) {
  if (!Array.isArray(vertices) || vertices.length < 2) {
    throw new Error("rasterizePolygon expects at least 2 vertices");
  }

  const out = [];
  const seen = new Set();

  function pushPoint(p) {
    const key = `${p[0]},${p[1]}`;
    if (!seen.has(key)) {
      seen.add(key);
      out.push(p);
    }
  }

  for (let i = 0; i < vertices.length; i++) {
    const [x0, y0] = vertices[i];
    const [x1, y1] = vertices[(i + 1) % vertices.length];
    const edgePoints = method(x0, y0, x1, y1);
    for (const p of edgePoints) pushPoint(p);
  }

  return out;
}

/**
 * Utility: render points into a text grid for quick inspection.
 */
function renderPoints(points, width, height, empty = ".", filled = "#") {
  const grid = Array.from({ length: height }, () => Array(width).fill(empty));

  for (const [x, y] of points) {
    if (y >= 0 && y < height && x >= 0 && x < width) {
      grid[y][x] = filled;
    }
  }

  return grid.map(row => row.join("")).join("\n");
}

/**
 * Demo
 */
function demo() {
  const x0 = 10, y0 = 20, x1 = 30, y1 = 50;

  const classical = bresenhamInteger(x0, y0, x1, y1);
  const algebraic = rasterizeImplicitLine(x0, y0, x1, y1);
  const cellBased = rasterizeImplicitByCells(x0, y0, x1, y1);

  console.log("Classical Bresenham count:", classical.length);
  console.log("Implicit algebraic count:", algebraic.length);
  console.log("Cell-based algebraic count:", cellBased.length);

  console.log("\nClassical first 10 points:", classical.slice(0, 10));
  console.log("Algebraic first 10 points:", algebraic.slice(0, 10));

  const poly = [
    [2, 2],
    [18, 4],
    [22, 14],
    [8, 18]
  ];

  const polyPoints = rasterizePolygon(poly, bresenhamInteger);
  console.log("\nPolygon rasterization:");
  console.log(renderPoints(polyPoints, 26, 22));
}

demo();

/**
 * Export for Node.js usage if desired.
 */
if (typeof module !== "undefined") {
  module.exports = {
    ImplicitLine,
    bresenhamInteger,
    rasterizeImplicitLine,
    rasterizeImplicitByCells,
    rasterizePolygon,
    renderPoints
  };
}

A few important notes about the design:

The ImplicitLine class is the algebraic core. It treats the line as a polynomial equation, which is the closest practical analogue to the “equation-first” style you were aiming for.

bresenhamInteger is the correct baseline implementation. Your original bresenhamAlgebraic idea had a conceptual issue: comparing two independent error updates is not enough to guarantee a proper path through all octants.

rasterizeImplicitLine is the “Grothendieck-inspired” variant: instead of thinking in terms of slope, it treats the line as a zero set and chooses lattice moves that stay closest to that zero set.

rasterizePolygon shows how the same algebraic-line machinery can be reused as a small tessellation engine for polygons.

If you want, I can turn this into a more ambitious version with:

anti-aliased coverage weights,

quadratic and cubic implicit curves,

triangle/mesh tessellation,

or a small browser demo with canvas rendering.
