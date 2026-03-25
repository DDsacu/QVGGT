document.addEventListener('DOMContentLoaded', () => {
  const cards = Array.from(document.querySelectorAll('.input-card'));
  const viewerPred = document.getElementById('modelViewerComparison1');
  const viewerRaw = document.getElementById('modelViewerComparison2');

  const selectCard = (card) => {
    cards.forEach((c) => c.classList.remove('is-active'));
    card.classList.add('is-active');

    const pred = card.getAttribute('data-pred');
    const raw = card.getAttribute('data-raw');

    if (viewerPred && pred) viewerPred.setAttribute('src', pred);
    if (viewerRaw && raw) viewerRaw.setAttribute('src', raw);
  };

  cards.forEach((card) => {
    card.addEventListener('click', () => selectCard(card));
  });

  if (cards[0]) selectCard(cards[0]);
});
