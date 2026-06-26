<script lang="ts">
  import { onMount, getContext } from 'svelte';
  import { gsap } from 'gsap';

  // Manim-style colors
  export let stroke = '#58C4DD'; // BLUE (Manim default)
  export let strokeWidth = 3;
  export let fill = 'none';
  export let d = ''; // SVG path data
  export let delay = 0;
  export let duration = 1;
  export let ease = 'power2.inOut';
  export let glow = false;

  // Animation type
  export let animation: 'create' | 'draw' | 'fadeIn' | 'none' = 'create';

  let pathEl: SVGPathElement;
  let pathLength = 0;

  onMount(() => {
    if (!pathEl) return;

    pathLength = pathEl.getTotalLength();

    if (animation === 'create' || animation === 'draw') {
      // Start with path hidden
      gsap.set(pathEl, {
        strokeDasharray: pathLength,
        strokeDashoffset: pathLength,
        opacity: 1
      });

      // Animate drawing
      gsap.to(pathEl, {
        strokeDashoffset: 0,
        duration,
        delay,
        ease
      });
    } else if (animation === 'fadeIn') {
      gsap.set(pathEl, { opacity: 0 });
      gsap.to(pathEl, {
        opacity: 1,
        duration,
        delay,
        ease
      });
    }
  });
</script>

<path
  bind:this={pathEl}
  {d}
  {stroke}
  stroke-width={strokeWidth}
  {fill}
  stroke-linecap="round"
  stroke-linejoin="round"
  filter={glow ? 'url(#glow)' : undefined}
  opacity={animation === 'none' ? 1 : 0}
/>
