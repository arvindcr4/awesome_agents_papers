<script lang="ts">
  import { onMount } from 'svelte';
  import { gsap } from 'gsap';

  export let x = 0;
  export let y = 0;
  export let text = '';
  export let fill = '#FFFFFF';
  export let fontSize = 24;
  export let fontFamily = 'CMU Serif, Georgia, serif';
  export let textAnchor: 'start' | 'middle' | 'end' = 'middle';
  export let delay = 0;
  export let duration = 0.5;
  export let ease = 'power2.out';
  export let animation: 'write' | 'fadeIn' | 'typewriter' | 'none' = 'write';
  export let glow = false;

  let textEl: SVGTextElement;
  let displayText = '';

  onMount(() => {
    if (!textEl) return;

    if (animation === 'write') {
      gsap.set(textEl, { opacity: 0, y: y + 10 });
      gsap.to(textEl, {
        opacity: 1,
        y,
        duration,
        delay,
        ease
      });
      displayText = text;
    } else if (animation === 'fadeIn') {
      gsap.set(textEl, { opacity: 0 });
      gsap.to(textEl, {
        opacity: 1,
        duration,
        delay,
        ease
      });
      displayText = text;
    } else if (animation === 'typewriter') {
      displayText = '';
      const chars = text.split('');
      const charDuration = duration / chars.length;

      chars.forEach((char, i) => {
        gsap.delayedCall(delay + i * charDuration, () => {
          displayText += char;
        });
      });
    } else {
      displayText = text;
    }
  });
</script>

<text
  bind:this={textEl}
  {x}
  {y}
  {fill}
  font-size={fontSize}
  font-family={fontFamily}
  text-anchor={textAnchor}
  dominant-baseline="middle"
  filter={glow ? 'url(#glow)' : undefined}
  opacity={animation === 'none' ? 1 : 0}
>
  {displayText || text}
</text>
