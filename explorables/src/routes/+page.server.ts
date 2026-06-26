import type { PageServerLoad } from './$types';
import fs from 'fs';
import path from 'path';

export const load: PageServerLoad = async () => {
  const dataDir = path.join(process.cwd(), 'src/lib/data/papers');
  const indexPath = path.join(dataDir, 'index.json');

  // Default categories
  const defaultCategories = [
    'agent-frameworks',
    'benchmarks',
    'computer-use',
    'memory-rag',
    'multi-agent',
    'planning',
    'reasoning',
    'rl-finetuning',
    'robotics',
    'security',
    'theorem-proving',
    'web-agents'
  ];

  try {
    if (fs.existsSync(indexPath)) {
      const content = fs.readFileSync(indexPath, 'utf-8');
      const data = JSON.parse(content);
      // Handle both array format and object format
      const papers = Array.isArray(data) ? data : (data.papers || []);
      const categories = Array.isArray(data) ? defaultCategories : (data.categories || defaultCategories);
      return { papers, categories };
    }

    // If no index.json, scan for individual paper files
    if (fs.existsSync(dataDir)) {
      const files = fs.readdirSync(dataDir).filter(f => f.endsWith('.json') && f !== 'index.json');
      const papers = files.map(file => {
        const content = fs.readFileSync(path.join(dataDir, file), 'utf-8');
        return JSON.parse(content);
      });
      return { papers, categories: defaultCategories };
    }

    return { papers: [], categories: defaultCategories };
  } catch (error) {
    console.error('Error loading papers:', error);
    return { papers: [], categories: defaultCategories };
  }
};
