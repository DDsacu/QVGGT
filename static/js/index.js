document.addEventListener('DOMContentLoaded', () => {
  const cards = Array.from(document.querySelectorAll('.input-card'));
  const viewerPred = document.getElementById('modelViewerComparison1');
  const viewerRaw = document.getElementById('modelViewerComparison2');
  const gallery = document.getElementById('inputGallery');
  const galleryImage = document.getElementById('galleryImage');
  const galleryCount = document.getElementById('galleryCount');
  const galleryPrev = document.querySelector('[data-gallery-prev]');
  const galleryNext = document.querySelector('[data-gallery-next]');
  let galleryItems = [];
  let galleryIndex = 0;

  document.querySelectorAll('.collage-cell img').forEach((image) => {
    image.addEventListener('error', () => {
      image.style.display = 'none';
    });
  });

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

  const renderGallery = () => {
    if (!gallery || !galleryImage || galleryItems.length === 0) return;

    galleryImage.style.display = '';
    galleryImage.setAttribute('src', galleryItems[galleryIndex]);

    const hasMultiple = galleryItems.length > 1;
    galleryPrev?.classList.toggle('is-hidden', !hasMultiple);
    galleryNext?.classList.toggle('is-hidden', !hasMultiple);
    galleryCount?.classList.toggle('is-hidden', !hasMultiple);

    if (galleryCount) {
      galleryCount.textContent = `${galleryIndex + 1} / ${galleryItems.length}`;
    }
  };

  const openGallery = (items, startIndex = 0) => {
    if (!gallery || !galleryImage || items.length === 0) return;

    galleryItems = items;
    galleryIndex = startIndex;
    gallery.classList.add('is-open');
    gallery.setAttribute('aria-hidden', 'false');
    document.body.style.overflow = 'hidden';
    renderGallery();
  };

  const closeGallery = () => {
    if (!gallery || !galleryImage) return;

    gallery.classList.remove('is-open');
    gallery.setAttribute('aria-hidden', 'true');
    galleryImage.removeAttribute('src');
    document.body.style.overflow = '';
  };

  const stepGallery = (direction) => {
    if (galleryItems.length < 2) return;
    galleryIndex = (galleryIndex + direction + galleryItems.length) % galleryItems.length;
    renderGallery();
  };

  document.querySelectorAll('.js-gallery-trigger').forEach((trigger) => {
    trigger.addEventListener('click', (event) => {
      event.preventDefault();
      event.stopPropagation();

      const card = trigger.closest('.input-card');
      if (card) selectCard(card);

      const items = (trigger.getAttribute('data-gallery') || '')
        .split(',')
        .map((item) => item.trim())
        .filter(Boolean);

      openGallery(items);
    });
  });

  document.querySelectorAll('[data-gallery-close]').forEach((closeButton) => {
    closeButton.addEventListener('click', closeGallery);
  });

  galleryPrev?.addEventListener('click', () => stepGallery(-1));
  galleryNext?.addEventListener('click', () => stepGallery(1));

  document.addEventListener('keydown', (event) => {
    if (!gallery?.classList.contains('is-open')) return;

    if (event.key === 'Escape') closeGallery();
    if (event.key === 'ArrowLeft') stepGallery(-1);
    if (event.key === 'ArrowRight') stepGallery(1);
  });
});
