This script generates a Logarithmic spiral ($r = ae^{b\theta}$) by scaling the parameters of the algebraic ellipse algorithm.

```javascript
/**
 * Generates a Logarithmic spiral.
 */
function drawLogarithmicSpiral(canvas) {
  const ctx = canvas.getContext("2d");
  const xc = canvas.width / 2;
  const yc = canvas.height / 2;

  const a = 2, b = 0.15;
  for (let theta = 0; theta < 8 * Math.PI; theta += 0.05) {
    const radius = a * Math.exp(b * theta);
    const x = xc + radius * Math.cos(theta);
    const y = yc + radius * Math.sin(theta);

    ctx.fillStyle = `hsl(${theta * 30}, 80%, 40%)`;
    ctx.beginPath();
    ctx.arc(x, y, 1.5, 0, 2 * Math.PI);
    ctx.fill();
  }
}
```

The logarithmic spiral, often found in nature (e.g., nautilus shells), demonstrates the interplay between exponential growth and rotation, mapped onto the integer lattice.
