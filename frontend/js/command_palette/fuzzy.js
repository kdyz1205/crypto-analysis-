// frontend/js/command_palette/fuzzy.js — lightweight subsequence fuzzy matcher

export function fuzzyMatch(query, text) {
  if (!query) return { score: 0, indexes: [] };
  const q = query.toLowerCase();
  const t = text.toLowerCase();
  let score = 0;
  let qi = 0;
  const indexes = [];
  for (let ti = 0; ti < t.length && qi < q.length; ti++) {
    if (t[ti] === q[qi]) {
      indexes.push(ti);
      score += 10;
      if (indexes.length > 1 && indexes[indexes.length - 1] === indexes[indexes.length - 2] + 1) {
        score += 5;
      }
      qi++;
    }
  }
  return qi === q.length ? { score, indexes } : null;
}

export function fuzzySort(query, items, textFn = (x) => x) {
  return items
    .map((item) => ({ item, match: fuzzyMatch(query, textFn(item)) }))
    .filter((x) => x.match)
    .sort((a, b) => b.match.score - a.match.score);
}
