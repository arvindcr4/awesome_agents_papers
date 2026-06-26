<script lang="ts">
  import { onMount, createEventDispatcher } from 'svelte';
  import { gsap } from 'gsap';

  export let width = 800;
  export let height = 450;
  export let background = '#1a1a2e';
  export let autoplay = true;

  let svg: SVGSVGElement;
  let timeline: gsap.core.Timeline;
  let isPlaying = false;

  const dispatch = createEventDispatcher();

  onMount(() => {
    timeline = gsap.timeline({
      paused: !autoplay,
      onComplete: () => {
        isPlaying = false;
        dispatch('complete');
      }
    });

    if (autoplay) {
      isPlaying = true;
    }

    dispatch('ready', { timeline, svg });
  });

  export function play() {
    timeline?.play();
    isPlaying = true;
  }

  export function pause() {
    timeline?.pause();
    isPlaying = false;
  }

  export function restart() {
    timeline?.restart();
    isPlaying = true;
  }

  export function getTimeline() {
    return timeline;
  }

  export function getSvg() {
    return svg;
  }
</script>

<div class="manim-scene">
  <svg
    bind:this={svg}
    {width}
    {height}
    viewBox="0 0 {width} {height}"
    style="background: {background}"
  >
    <defs>
      <!-- Glow filter for Manim-style effects -->
      <filter id="glow" x="-50%" y="-50%" width="200%" height="200%">
        <feGaussianBlur stdDeviation="2" result="coloredBlur"/>
        <feMerge>
          <feMergeNode in="coloredBlur"/>
          <feMergeNode in="SourceGraphic"/>
        </feMerge>
      </filter>

      <!-- Arrow marker -->
      <marker id="arrowhead" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
        <polygon points="0 0, 10 3.5, 0 7" fill="currentColor"/>
      </marker>
    </defs>

    <slot {timeline} {svg} />
  </svg>

  <div class="controls">
    {#if !isPlaying}
      <button on:click={play} aria-label="Play">
        <svg viewBox="0 0 24 24" width="24" height="24">
          <path fill="currentColor" d="M8 5v14l11-7z"/>
        </svg>
      </button>
    {:else}
      <button on:click={pause} aria-label="Pause">
        <svg viewBox="0 0 24 24" width="24" height="24">
          <path fill="currentColor" d="M6 19h4V5H6v14zm8-14v14h4V5h-4z"/>
        </svg>
      </button>
    {/if}
    <button on:click={restart} aria-label="Restart">
      <svg viewBox="0 0 24 24" width="24" height="24">
        <path fill="currentColor" d="M12 5V1L7 6l5 5V7c3.31 0 6 2.69 6 6s-2.69 6-6 6-6-2.69-6-6H4c0 4.42 3.58 8 8 8s8-3.58 8-8-3.58-8-8-8z"/>
      </svg>
    </button>
  </div>
</div>

<style>
  .manim-scene {
    position: relative;
    display: inline-block;
    border-radius: 8px;
    overflow: hidden;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
  }

  svg {
    display: block;
  }

  .controls {
    position: absolute;
    bottom: 10px;
    left: 50%;
    transform: translateX(-50%);
    display: flex;
    gap: 8px;
    opacity: 0;
    transition: opacity 0.2s;
  }

  .manim-scene:hover .controls {
    opacity: 1;
  }

  button {
    background: rgba(255, 255, 255, 0.15);
    border: none;
    border-radius: 50%;
    width: 36px;
    height: 36px;
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
    color: white;
    transition: background 0.2s;
  }

  button:hover {
    background: rgba(255, 255, 255, 0.25);
  }
</style>
