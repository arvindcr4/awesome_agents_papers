<script lang="ts">
  /**
   * Step-through animation controls for algorithms/processes.
   * Play/pause/step interface with progress indicator.
   */
  import { createEventDispatcher, onDestroy } from 'svelte';

  export let steps: string[] = [];
  export let currentStep: number = 0;
  export let autoplayInterval: number = 1500;

  const dispatch = createEventDispatcher();

  let playing = false;
  let intervalId: ReturnType<typeof setInterval> | null = null;

  function play() {
    if (playing) return;
    playing = true;
    intervalId = setInterval(() => {
      if (currentStep < steps.length - 1) {
        currentStep++;
        dispatch('step', currentStep);
      } else {
        pause();
      }
    }, autoplayInterval);
  }

  function pause() {
    playing = false;
    if (intervalId) {
      clearInterval(intervalId);
      intervalId = null;
    }
  }

  function next() {
    if (currentStep < steps.length - 1) {
      currentStep++;
      dispatch('step', currentStep);
    }
  }

  function prev() {
    if (currentStep > 0) {
      currentStep--;
      dispatch('step', currentStep);
    }
  }

  function reset() {
    pause();
    currentStep = 0;
    dispatch('step', currentStep);
  }

  function goToStep(index: number) {
    currentStep = index;
    dispatch('step', currentStep);
  }

  onDestroy(() => {
    if (intervalId) clearInterval(intervalId);
  });
</script>

<div class="step-controls">
  <div class="buttons">
    <button on:click={reset} title="Reset" disabled={currentStep === 0 && !playing}>
      ⟲
    </button>
    <button on:click={prev} title="Previous" disabled={currentStep === 0}>
      ◀
    </button>
    {#if playing}
      <button on:click={pause} title="Pause" class="play-pause">
        ⏸
      </button>
    {:else}
      <button on:click={play} title="Play" class="play-pause" disabled={currentStep >= steps.length - 1}>
        ▶
      </button>
    {/if}
    <button on:click={next} title="Next" disabled={currentStep >= steps.length - 1}>
      ▶
    </button>
  </div>

  <div class="progress">
    {#each steps as step, i}
      <button
        class="step-dot"
        class:active={i === currentStep}
        class:completed={i < currentStep}
        on:click={() => goToStep(i)}
        title={step}
      />
    {/each}
  </div>

  <div class="step-label">
    <span class="step-number">Step {currentStep + 1}/{steps.length}</span>
    <span class="step-name">{steps[currentStep]}</span>
  </div>
</div>

<style>
  .step-controls {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 0.75rem;
    padding: 1rem;
    background: #f9fafb;
    border-radius: 8px;
    margin: 1rem 0;
  }

  .buttons {
    display: flex;
    gap: 0.5rem;
  }

  button {
    width: 36px;
    height: 36px;
    border: none;
    border-radius: 50%;
    background: white;
    color: #374151;
    font-size: 1rem;
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
    transition: all 0.15s;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
  }

  button:hover:not(:disabled) {
    background: #3b82f6;
    color: white;
  }

  button:disabled {
    opacity: 0.4;
    cursor: not-allowed;
  }

  .play-pause {
    width: 44px;
    height: 44px;
    font-size: 1.2rem;
    background: #3b82f6;
    color: white;
  }

  .play-pause:hover:not(:disabled) {
    background: #2563eb;
  }

  .progress {
    display: flex;
    gap: 0.5rem;
  }

  .step-dot {
    width: 10px;
    height: 10px;
    padding: 0;
    background: #d1d5db;
    box-shadow: none;
  }

  .step-dot.completed {
    background: #10b981;
  }

  .step-dot.active {
    background: #3b82f6;
    transform: scale(1.3);
  }

  .step-label {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 0.25rem;
  }

  .step-number {
    font-size: 0.75rem;
    color: #9ca3af;
    font-family: 'SF Mono', Monaco, monospace;
  }

  .step-name {
    font-size: 0.9rem;
    color: #374151;
    font-weight: 500;
  }
</style>
