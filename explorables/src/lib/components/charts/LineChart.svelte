<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import * as d3 from 'd3';
  import { hoverState, setHover, clearHover } from '$lib/stores/hoverState';

  interface DataPoint {
    x: number;
    y: number;
    label?: string;
  }

  interface Series {
    name: string;
    data: DataPoint[];
    color: string;
  }

  export let series: Series[] = [];
  export let width = 600;
  export let height = 300;
  export let xLabel = '';
  export let yLabel = '';
  export let title = '';
  export let chartId = 'chart';
  export let showGrid = true;
  export let logScale = false;
  export let annotations: { x: number; text: string }[] = [];

  let container: HTMLDivElement;
  let svg: d3.Selection<SVGSVGElement, unknown, null, undefined>;

  const margin = { top: 40, right: 120, bottom: 50, left: 60 };
  const innerWidth = width - margin.left - margin.right;
  const innerHeight = height - margin.top - margin.bottom;

  let unsubscribe: () => void;

  onMount(() => {
    createChart();

    unsubscribe = hoverState.subscribe(state => {
      if (svg && state.source !== chartId) {
        updateHoverLine(state.x);
      }
    });
  });

  onDestroy(() => {
    if (unsubscribe) unsubscribe();
  });

  function createChart() {
    // Clear existing
    d3.select(container).selectAll('*').remove();

    svg = d3.select(container)
      .append('svg')
      .attr('width', width)
      .attr('height', height)
      .attr('class', 'line-chart');

    const g = svg.append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`);

    // Scales
    const allData = series.flatMap(s => s.data);
    const xExtent = d3.extent(allData, d => d.x) as [number, number];
    const yExtent = d3.extent(allData, d => d.y) as [number, number];

    const xScale = d3.scaleLinear()
      .domain(xExtent)
      .range([0, innerWidth]);

    const yScale = logScale
      ? d3.scaleLog().domain([Math.max(yExtent[0], 1e-7), yExtent[1]]).range([innerHeight, 0])
      : d3.scaleLinear().domain(yExtent).range([innerHeight, 0]).nice();

    // Grid
    if (showGrid) {
      g.append('g')
        .attr('class', 'grid')
        .attr('opacity', 0.1)
        .call(d3.axisLeft(yScale).tickSize(-innerWidth).tickFormat(() => ''));
    }

    // Axes
    g.append('g')
      .attr('transform', `translate(0,${innerHeight})`)
      .call(d3.axisBottom(xScale).ticks(6))
      .attr('color', '#666');

    g.append('g')
      .call(d3.axisLeft(yScale).ticks(5))
      .attr('color', '#666');

    // Axis labels
    if (xLabel) {
      g.append('text')
        .attr('x', innerWidth / 2)
        .attr('y', innerHeight + 40)
        .attr('text-anchor', 'middle')
        .attr('fill', '#666')
        .attr('font-size', '12px')
        .text(xLabel);
    }

    if (yLabel) {
      g.append('text')
        .attr('transform', 'rotate(-90)')
        .attr('x', -innerHeight / 2)
        .attr('y', -45)
        .attr('text-anchor', 'middle')
        .attr('fill', '#666')
        .attr('font-size', '12px')
        .text(yLabel);
    }

    // Title
    if (title) {
      svg.append('text')
        .attr('x', width / 2)
        .attr('y', 20)
        .attr('text-anchor', 'middle')
        .attr('font-weight', '500')
        .attr('font-size', '14px')
        .text(title);
    }

    // Line generator
    const line = d3.line<DataPoint>()
      .x(d => xScale(d.x))
      .y(d => yScale(d.y))
      .curve(d3.curveMonotoneX);

    // Draw lines
    series.forEach(s => {
      g.append('path')
        .datum(s.data)
        .attr('fill', 'none')
        .attr('stroke', s.color)
        .attr('stroke-width', 2)
        .attr('d', line);
    });

    // Legend
    const legend = svg.append('g')
      .attr('transform', `translate(${width - margin.right + 10}, ${margin.top})`);

    series.forEach((s, i) => {
      const lg = legend.append('g')
        .attr('transform', `translate(0, ${i * 20})`);

      lg.append('line')
        .attr('x1', 0)
        .attr('x2', 20)
        .attr('y1', 0)
        .attr('y2', 0)
        .attr('stroke', s.color)
        .attr('stroke-width', 2);

      lg.append('text')
        .attr('x', 25)
        .attr('y', 4)
        .attr('font-size', '11px')
        .attr('fill', '#333')
        .text(s.name);
    });

    // Hover line (initially hidden)
    const hoverLine = g.append('line')
      .attr('class', 'hover-line')
      .attr('y1', 0)
      .attr('y2', innerHeight)
      .attr('stroke', '#333')
      .attr('stroke-width', 1)
      .attr('stroke-dasharray', '4,4')
      .attr('opacity', 0);

    // Hover value text
    const hoverText = g.append('text')
      .attr('class', 'hover-text')
      .attr('font-size', '11px')
      .attr('fill', '#333')
      .attr('opacity', 0);

    // Hover dots
    const hoverDots = g.append('g').attr('class', 'hover-dots');

    // Annotations
    annotations.forEach(ann => {
      const x = xScale(ann.x);
      g.append('line')
        .attr('x1', x)
        .attr('x2', x)
        .attr('y1', 0)
        .attr('y2', innerHeight)
        .attr('stroke', '#999')
        .attr('stroke-dasharray', '2,2');

      g.append('text')
        .attr('x', x)
        .attr('y', -5)
        .attr('text-anchor', 'middle')
        .attr('font-size', '10px')
        .attr('fill', '#666')
        .text(ann.text);
    });

    // Invisible overlay for mouse events
    const bisect = d3.bisector<DataPoint, number>(d => d.x).left;

    g.append('rect')
      .attr('width', innerWidth)
      .attr('height', innerHeight)
      .attr('fill', 'transparent')
      .on('mousemove', (event) => {
        const [mouseX] = d3.pointer(event);
        const x0 = xScale.invert(mouseX);

        // Find closest data point
        const firstSeries = series[0].data;
        const i = bisect(firstSeries, x0, 1);
        const d0 = firstSeries[i - 1];
        const d1 = firstSeries[i];
        const closest = d1 && (x0 - d0.x > d1.x - x0) ? d1 : d0;

        if (closest) {
          const xPos = xScale(closest.x);
          setHover(closest.x, i, chartId);

          // Update hover elements
          hoverLine
            .attr('x1', xPos)
            .attr('x2', xPos)
            .attr('opacity', 1);

          hoverText
            .attr('x', xPos + 5)
            .attr('y', -5)
            .text(`x: ${closest.x.toLocaleString()}`)
            .attr('opacity', 1);

          // Update dots
          hoverDots.selectAll('circle').remove();
          series.forEach(s => {
            const idx = bisect(s.data, closest.x, 1);
            const point = s.data[idx - 1] || s.data[idx];
            if (point) {
              hoverDots.append('circle')
                .attr('cx', xScale(point.x))
                .attr('cy', yScale(point.y))
                .attr('r', 5)
                .attr('fill', s.color)
                .attr('stroke', 'white')
                .attr('stroke-width', 2);
            }
          });
        }
      })
      .on('mouseleave', () => {
        clearHover();
        hoverLine.attr('opacity', 0);
        hoverText.attr('opacity', 0);
        hoverDots.selectAll('circle').remove();
      });
  }

  function updateHoverLine(xValue: number | null) {
    if (!svg) return;

    const g = svg.select('g');
    const hoverLine = g.select('.hover-line');
    const hoverDots = g.select('.hover-dots');

    if (xValue === null) {
      hoverLine.attr('opacity', 0);
      hoverDots.selectAll('circle').remove();
      return;
    }

    const allData = series.flatMap(s => s.data);
    const xExtent = d3.extent(allData, d => d.x) as [number, number];
    const xScale = d3.scaleLinear().domain(xExtent).range([0, innerWidth]);
    const yExtent = d3.extent(allData, d => d.y) as [number, number];
    const yScale = logScale
      ? d3.scaleLog().domain([Math.max(yExtent[0], 1e-7), yExtent[1]]).range([innerHeight, 0])
      : d3.scaleLinear().domain(yExtent).range([innerHeight, 0]).nice();

    const xPos = xScale(xValue);
    hoverLine
      .attr('x1', xPos)
      .attr('x2', xPos)
      .attr('opacity', 1);

    // Update dots
    const bisect = d3.bisector<DataPoint, number>(d => d.x).left;
    hoverDots.selectAll('circle').remove();
    series.forEach(s => {
      const idx = bisect(s.data, xValue, 1);
      const point = s.data[idx - 1] || s.data[idx];
      if (point) {
        hoverDots.append('circle')
          .attr('cx', xScale(point.x))
          .attr('cy', yScale(point.y))
          .attr('r', 5)
          .attr('fill', s.color)
          .attr('stroke', 'white')
          .attr('stroke-width', 2);
      }
    });
  }

  $: if (series.length && container) {
    createChart();
  }
</script>

<div bind:this={container} class="chart-container"></div>

<style>
  .chart-container {
    display: inline-block;
  }

  :global(.line-chart) {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
  }
</style>
