import { writable, derived } from 'svelte/store';

// Shared hover state for synchronized charts
export interface HoverState {
  x: number | null;
  y: number | null;
  dataIndex: number | null;
  source: string | null;
}

export const hoverState = writable<HoverState>({
  x: null,
  y: null,
  dataIndex: null,
  source: null
});

export const isHovering = derived(hoverState, ($state) => $state.x !== null);

export function setHover(x: number, dataIndex: number, source: string) {
  hoverState.set({ x, y: null, dataIndex, source });
}

export function clearHover() {
  hoverState.set({ x: null, y: null, dataIndex: null, source: null });
}
