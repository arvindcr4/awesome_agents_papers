<script lang="ts">
  import { onMount } from 'svelte';
  import { gsap } from 'gsap';

  export let cx = 0;
  export let cy = 0;
  export let r = 50;
  export let stroke = '#FFFF00'; // YELLOW (Manim)
  export let strokeWidth = 3;
  export let fill = 'none';
  export let delay = 0;
  export let duration = 1;
  export let ease = 'power2.inOut';
  export let animation: 'create' | 'growFromCenter' | 'fadeIn' | 'none' = 'create';
  export let glow = false;

  let circleEl: SVGCircleElement;

  onMount(() => {
    if (!circleEl) return;

    const circumference = 2 * Math.PI * r;

    if (animation === 'create') {
      gsap.set(circleEl, {
        strokeDasharray: circumference,
        strokeDashoffset: circumference,
        opacity: 1
      });

      gsap.to(circleEl, {
        strokeDashoffset: 0,
        duration,
        delay,
        ease
      });
    } else if (animation === 'growFromCenter') {
      gsap.set(circleEl, {
        attr: { r: 0 },
        opacity: 1
      });

      gsap.to(circleEl, {
        attr: { r },
        duration,
        delay,
        ease: 'back.out(1.7)'
      });
    } else if (animation === 'fadeIn') {
      gsap.set(circleEl, { opacity: 0 });
      gsap.to(circleEl, {
        opacity: 1,
        duration,
        delay,
        ease
      });
    }
  });
</script>

<circle
  bind:this={circleEl}
  {cx}
  {cy}
  {r}
  {stroke}
  stroke-width={strokeWidth}
  {fill}
  filter={glow ? 'url(#glow)' : undefined}
  opacity={animation === 'none' ? 1 : 0}
/>
