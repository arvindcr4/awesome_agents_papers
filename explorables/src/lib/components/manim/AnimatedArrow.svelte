<script lang="ts">
  import { onMount } from 'svelte';
  import { gsap } from 'gsap';

  export let x1 = 0;
  export let y1 = 0;
  export let x2 = 100;
  export let y2 = 0;
  export let stroke = '#FC6255'; // RED (Manim)
  export let strokeWidth = 3;
  export let delay = 0;
  export let duration = 0.8;
  export let ease = 'power2.inOut';
  export let animation: 'growArrow' | 'fadeIn' | 'none' = 'growArrow';
  export let glow = false;
  export let tipSize = 12;

  let lineEl: SVGLineElement;
  let tipEl: SVGPolygonElement;

  $: angle = Math.atan2(y2 - y1, x2 - x1) * (180 / Math.PI);
  $: tipPoints = `0,0 ${-tipSize},${-tipSize/2} ${-tipSize},${tipSize/2}`;

  onMount(() => {
    if (!lineEl || !tipEl) return;

    if (animation === 'growArrow') {
      // Start with line at origin point
      gsap.set(lineEl, {
        attr: { x2: x1, y2: y1 },
        opacity: 1
      });
      gsap.set(tipEl, {
        opacity: 0,
        attr: { transform: `translate(${x1},${y1}) rotate(${angle})` }
      });

      // Animate line growing
      gsap.to(lineEl, {
        attr: { x2, y2 },
        duration,
        delay,
        ease
      });

      // Animate tip following
      gsap.to(tipEl, {
        opacity: 1,
        attr: { transform: `translate(${x2},${y2}) rotate(${angle})` },
        duration,
        delay,
        ease
      });
    } else if (animation === 'fadeIn') {
      gsap.set([lineEl, tipEl], { opacity: 0 });
      gsap.to([lineEl, tipEl], {
        opacity: 1,
        duration,
        delay,
        ease
      });
    }
  });
</script>

<g filter={glow ? 'url(#glow)' : undefined}>
  <line
    bind:this={lineEl}
    {x1}
    {y1}
    {x2}
    {y2}
    {stroke}
    stroke-width={strokeWidth}
    stroke-linecap="round"
    opacity={animation === 'none' ? 1 : 0}
  />
  <polygon
    bind:this={tipEl}
    points={tipPoints}
    fill={stroke}
    transform="translate({x2},{y2}) rotate({angle})"
    opacity={animation === 'none' ? 1 : 0}
  />
</g>
