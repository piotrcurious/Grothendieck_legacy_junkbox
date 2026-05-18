This algorithm views the integer lattice as a **Base Scheme** $\text{Spec}(\mathbb{Z})$ and the line as an algebraic variety. We "reinvent" Bresenham by evaluating the local valuation (the error term) to find the best integer approximation of a "rational point" on the line.

```javascript
/**
 * Represents a line as an algebraic variety defined by ax + by + c = 0.
 */
function LineVariety(x1, y1, x2, y2) {
  this.a = y1 - y2;
  this.b = x2 - x1;
  this.c = x1 * y2 - x2 * y1;
}

/**
 * Finding Rational Points on the Base Scheme Spec(Z)
 */
function bresenhamAlgebraic(x0, y0, x1, y1) {
  const variety = new LineVariety(x0, y0, x1, y1);
  const points = [];

  let x = x0, y = y0;
  const sx = (x0 < x1) ? 1 : -1;
  const sy = (y0 < y1) ? 1 : -1;

  // The valuation f(x, y) = ax + by + c
  let valuation = 0;

  while (true) {
    points.push([x, y]);
    if (x === x1 && y === y1) break;

    // Test the fiber of the morphism at potential next points
    const vX = valuation + variety.a * sx;
    const vY = valuation + variety.b * sy;

    // Choose the point that minimizes the absolute valuation
    if (Math.abs(vX) < Math.abs(vY)) {
        x += sx;
        valuation = vX;
    } else {
        y += sy;
        valuation = vY;
    }
  }
  return points;
}

const start = [10, 20], end = [30, 50];
const rationalPoints = bresenhamAlgebraic(start[0], start[1], end[0], end[1]);
console.log(`Found ${rationalPoints.length} points on the variety.`);
```

By framing the line as a variety and the pixels as points on a scheme, we unify classical graphics with modern algebraic geometry.
