import * as THREE from 'https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js';
import { OrbitControls } from 'https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/controls/OrbitControls.js';
import { GLTFLoader } from 'https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/loaders/GLTFLoader.js';

const loader = new GLTFLoader();
const viewers = new Map();

const setStatus = (container, message, hidden = false) => {
  const wrapper = container.closest('.model-wrapper-comparison');
  const status = wrapper?.querySelector('[data-viewer-status]');
  if (!status) return;

  status.textContent = message;
  status.classList.toggle('is-hidden', hidden);
};

const fitCameraToObject = (state, object) => {
  const box = new THREE.Box3().setFromObject(object);
  const size = box.getSize(new THREE.Vector3());
  const center = box.getCenter(new THREE.Vector3());
  const maxDim = Math.max(size.x, size.y, size.z) || 1;
  const distance = maxDim * 1.35;

  state.controls.target.copy(center);
  state.camera.position.set(center.x, center.y - distance, center.z + distance * 0.55);
  state.camera.near = Math.max(distance / 1000, 0.001);
  state.camera.far = distance * 100;
  state.camera.updateProjectionMatrix();
  state.controls.update();
};

const resizeViewer = (state) => {
  const width = state.container.clientWidth;
  const height = state.container.clientHeight;
  if (!width || !height) return;

  state.renderer.setSize(width, height, false);
  state.camera.aspect = width / height;
  state.camera.updateProjectionMatrix();
};

const disposeObject = (object) => {
  object.traverse((child) => {
    if (child.geometry) child.geometry.dispose();
    if (child.material) {
      const materials = Array.isArray(child.material) ? child.material : [child.material];
      materials.forEach((material) => material.dispose());
    }
  });
};

const loadGlb = (state, src) => {
  setStatus(state.container, 'Loading GLB...');

  loader.load(
    src,
    (gltf) => {
      if (state.current) {
        state.scene.remove(state.current);
        disposeObject(state.current);
      }

      state.current = gltf.scene;
      state.scene.add(state.current);
      fitCameraToObject(state, state.current);
      setStatus(state.container, '', true);
    },
    undefined,
    (error) => {
      setStatus(state.container, `GLB failed to load: ${src}${error?.message ? ` (${error.message})` : ''}`);
    }
  );
};

const initViewer = (container) => {
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0xffffff);

  const camera = new THREE.PerspectiveCamera(35, 1, 0.001, 10000);
  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  container.appendChild(renderer.domElement);

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.autoRotate = false;

  scene.add(new THREE.HemisphereLight(0xffffff, 0xb8c7d9, 2.2));
  const light = new THREE.DirectionalLight(0xffffff, 1.2);
  light.position.set(2, -3, 4);
  scene.add(light);

  const state = { container, scene, camera, renderer, controls, current: null };
  viewers.set(container.id, state);

  const animate = () => {
    controls.update();
    renderer.render(scene, camera);
    requestAnimationFrame(animate);
  };

  resizeViewer(state);
  animate();

  const src = container.getAttribute('data-src');
  if (src) loadGlb(state, src);
};

window.qvggtSetGlbViewerSource = (id, src) => {
  const state = viewers.get(id);
  if (state) loadGlb(state, src);
};

window.addEventListener('resize', () => {
  viewers.forEach(resizeViewer);
});

const initAllViewers = () => {
  document.querySelectorAll('.glb-viewer').forEach(initViewer);
};

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', initAllViewers);
} else {
  initAllViewers();
}
