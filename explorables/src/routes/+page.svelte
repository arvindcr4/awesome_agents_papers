<script lang="ts">
  import { ManimScene, AnimatedCircle, AnimatedText, AnimatedArrow, AnimatedPath, COLORS } from '$lib/components/manim';
  import type { PageData } from './$types';

  export let data: PageData;

  const categories = data.categories || [];
  const papers = data.papers || [];

  // Group papers by category
  $: papersByCategory = categories.reduce((acc: Record<string, typeof papers>, cat: string) => {
    acc[cat] = papers.filter((p: any) => p.category === cat);
    return acc;
  }, {} as Record<string, typeof papers>);

  const categoryDisplay: Record<string, string> = {
    'agent-frameworks': 'Agent Frameworks',
    'benchmarks': 'Benchmarks',
    'computer-use': 'Computer Use',
    'memory-rag': 'Memory & RAG',
    'multi-agent': 'Multi-Agent',
    'planning': 'Planning',
    'reasoning': 'Reasoning',
    'rl-finetuning': 'RL Fine-tuning',
    'robotics': 'Robotics',
    'security': 'Security',
    'theorem-proving': 'Theorem Proving',
    'web-agents': 'Web Agents'
  };

  const categoryColors: Record<string, string> = {
    'agent-frameworks': '#3b82f6',
    'benchmarks': '#10b981',
    'computer-use': '#f59e0b',
    'memory-rag': '#8b5cf6',
    'multi-agent': '#ec4899',
    'planning': '#06b6d4',
    'reasoning': '#6366f1',
    'rl-finetuning': '#f97316',
    'robotics': '#14b8a6',
    'security': '#ef4444',
    'theorem-proving': '#a855f7',
    'web-agents': '#84cc16'
  };
</script>

<svelte:head>
  <title>AI Agents Paper Explorables</title>
  <meta name="description" content="Interactive visual explanations of 88 AI agent research papers in the style of Distill.pub">
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Source+Serif+4:ital,wght@0,400;0,600;1,400&display=swap" rel="stylesheet">
</svelte:head>

<main>
  <header>
    <div class="hero-animation">
      <ManimScene width={800} height={200} autoplay={true}>
        <!-- Animated title -->
        <AnimatedText x={400} y={60} text="AI Agents" fill={COLORS.WHITE} fontSize={42} delay={0} />
        <AnimatedText x={400} y={110} text="Paper Explorables" fill={COLORS.TEAL} fontSize={32} delay={0.3} />

        <!-- Decorative nodes -->
        <AnimatedCircle cx={100} cy={100} r={15} stroke={COLORS.BLUE} delay={0.5} />
        <AnimatedCircle cx={700} cy={100} r={15} stroke={COLORS.GREEN} delay={0.6} />
        <AnimatedArrow x1={115} y1={100} x2={280} y2={85} stroke={COLORS.GREY} strokeWidth={1} delay={0.8} />
        <AnimatedArrow x1={520} y1={85} x2={685} y2={100} stroke={COLORS.GREY} strokeWidth={1} delay={0.9} />

        <!-- Animated dots -->
        {#each [150, 200, 250] as x, i}
          <AnimatedCircle cx={x} cy={160} r={4} stroke={COLORS.PURPLE} delay={1 + i * 0.1} />
        {/each}
        {#each [550, 600, 650] as x, i}
          <AnimatedCircle cx={x} cy={160} r={4} stroke={COLORS.PURPLE} delay={1.3 + i * 0.1} />
        {/each}
      </ManimScene>
    </div>

    <p class="tagline">
      Interactive visual explanations of AI agent research papers.
      <br>
      <span class="style-note">Inspired by <a href="https://distill.pub" target="_blank" rel="noopener">Distill.pub</a> and <a href="https://pair.withgoogle.com/explorables" target="_blank" rel="noopener">PAIR Explorables</a>.</span>
    </p>

    <div class="stats">
      <div class="stat">
        <span class="stat-number">{papers.length}</span>
        <span class="stat-label">Papers</span>
      </div>
      <div class="stat">
        <span class="stat-number">{categories.length}</span>
        <span class="stat-label">Categories</span>
      </div>
      <div class="stat">
        <span class="stat-number">∞</span>
        <span class="stat-label">Interactions</span>
      </div>
    </div>
  </header>

  {#if papers.length === 0}
    <section class="empty-state">
      <h2>Generating Explorables...</h2>
      <p>
        Run the generation script to create interactive explorables from the 88 AI agent papers:
      </p>
      <pre><code>cd scripts && python generate_explorables.py</code></pre>
      <p class="hint">
        This will use Groq + Kimi K2 to generate Distill.pub-style explanations with suggested visualizations.
      </p>
    </section>
  {:else}
    {#each categories as category}
      {#if papersByCategory[category]?.length > 0}
        <section class="category-section">
          <h2>
            <span class="category-dot" style="background: {categoryColors[category]}"></span>
            {categoryDisplay[category] || category}
            <span class="count">({papersByCategory[category].length})</span>
          </h2>

          <div class="paper-grid">
            {#each papersByCategory[category] as paper}
              <a href="/paper/{paper.slug}" class="paper-card">
                <div class="card-category" style="background: {categoryColors[category]}">{categoryDisplay[category]}</div>
                <h3>{paper.title}</h3>
                <p class="authors">{paper.authors?.slice(0, 3).join(', ')}{paper.authors?.length > 3 ? ' et al.' : ''}</p>
                <p class="abstract">{paper.abstract?.slice(0, 150)}...</p>
                <div class="card-footer">
                  <span class="year">{paper.year}</span>
                  {#if paper.arxivId}
                    <span class="arxiv">arXiv:{paper.arxivId}</span>
                  {/if}
                </div>
              </a>
            {/each}
          </div>
        </section>
      {/if}
    {/each}
  {/if}

  <footer>
    <p>
      Built with <a href="https://svelte.dev" target="_blank" rel="noopener">SvelteKit</a>,
      <a href="https://d3js.org" target="_blank" rel="noopener">D3.js</a>, and
      <a href="https://gsap.com" target="_blank" rel="noopener">GSAP</a>.
    </p>
    <p class="source">
      <a href="https://github.com/arvind/awesome_agents_papers" target="_blank" rel="noopener">View Source →</a>
    </p>
  </footer>
</main>

<style>
  :global(body) {
    margin: 0;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    background: #fafafa;
    color: #333;
  }

  main {
    max-width: 1200px;
    margin: 0 auto;
    padding: 2rem;
  }

  header {
    text-align: center;
    margin-bottom: 4rem;
  }

  .hero-animation {
    display: flex;
    justify-content: center;
    margin-bottom: 1.5rem;
  }

  .tagline {
    font-family: 'Source Serif 4', Georgia, serif;
    font-size: 1.25rem;
    color: #4b5563;
    margin-bottom: 2rem;
    line-height: 1.6;
  }

  .style-note {
    font-size: 0.95rem;
    color: #6b7280;
  }

  .style-note a {
    color: #3b82f6;
    text-decoration: none;
  }

  .style-note a:hover {
    text-decoration: underline;
  }

  .stats {
    display: flex;
    justify-content: center;
    gap: 3rem;
  }

  .stat {
    display: flex;
    flex-direction: column;
    align-items: center;
  }

  .stat-number {
    font-size: 2.5rem;
    font-weight: 600;
    color: #1f2937;
    font-family: 'Source Serif 4', Georgia, serif;
  }

  .stat-label {
    font-size: 0.85rem;
    color: #6b7280;
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }

  .empty-state {
    text-align: center;
    padding: 4rem 2rem;
    background: #f3f4f6;
    border-radius: 12px;
    margin: 2rem 0;
  }

  .empty-state h2 {
    font-family: 'Source Serif 4', Georgia, serif;
    font-size: 1.5rem;
    margin-bottom: 1rem;
  }

  .empty-state pre {
    background: #1f2937;
    color: #f9fafb;
    padding: 1rem 2rem;
    border-radius: 8px;
    display: inline-block;
    font-size: 0.9rem;
  }

  .hint {
    font-size: 0.9rem;
    color: #6b7280;
    margin-top: 1.5rem;
  }

  .category-section {
    margin-bottom: 4rem;
  }

  .category-section h2 {
    font-family: 'Source Serif 4', Georgia, serif;
    font-size: 1.5rem;
    font-weight: 600;
    margin-bottom: 1.5rem;
    display: flex;
    align-items: center;
    gap: 0.75rem;
  }

  .category-dot {
    width: 12px;
    height: 12px;
    border-radius: 50%;
  }

  .count {
    font-weight: 400;
    color: #6b7280;
    font-size: 1rem;
  }

  .paper-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
    gap: 1.5rem;
  }

  .paper-card {
    background: white;
    border-radius: 12px;
    padding: 1.5rem;
    text-decoration: none;
    color: inherit;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    transition: transform 0.2s, box-shadow 0.2s;
    display: flex;
    flex-direction: column;
  }

  .paper-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 8px 25px rgba(0, 0, 0, 0.12);
  }

  .card-category {
    display: inline-block;
    padding: 0.2rem 0.6rem;
    border-radius: 9999px;
    font-size: 0.7rem;
    font-weight: 500;
    color: white;
    text-transform: uppercase;
    letter-spacing: 0.03em;
    margin-bottom: 0.75rem;
    align-self: flex-start;
  }

  .paper-card h3 {
    font-family: 'Source Serif 4', Georgia, serif;
    font-size: 1.1rem;
    font-weight: 600;
    line-height: 1.3;
    margin-bottom: 0.5rem;
    color: #1f2937;
  }

  .authors {
    font-size: 0.85rem;
    color: #6b7280;
    margin-bottom: 0.75rem;
  }

  .abstract {
    font-size: 0.9rem;
    color: #4b5563;
    line-height: 1.5;
    flex-grow: 1;
    margin-bottom: 1rem;
  }

  .card-footer {
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 0.8rem;
    color: #9ca3af;
    border-top: 1px solid #f3f4f6;
    padding-top: 0.75rem;
    margin-top: auto;
  }

  .arxiv {
    font-family: monospace;
    font-size: 0.75rem;
  }

  footer {
    margin-top: 4rem;
    padding-top: 2rem;
    border-top: 1px solid #e5e7eb;
    text-align: center;
    color: #6b7280;
    font-size: 0.9rem;
  }

  footer a {
    color: #3b82f6;
    text-decoration: none;
  }

  footer a:hover {
    text-decoration: underline;
  }

  .source {
    margin-top: 0.5rem;
  }

  @media (max-width: 768px) {
    .stats {
      gap: 2rem;
    }

    .stat-number {
      font-size: 2rem;
    }

    .paper-grid {
      grid-template-columns: 1fr;
    }
  }
</style>
