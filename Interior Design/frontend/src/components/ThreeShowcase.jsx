import { useEffect, useRef } from "react";

// Displays the provided image as a textured plane with interactive 3D controls
// Requires 'three' dependency (already installed)

const ThreeShowcase = ({ imageUrl, width = 680, height = 420, isDarkMode = false }) => {
  const containerRef = useRef(null);
  const stateRef = useRef({ disposed: false });

  useEffect(() => {
    if (!imageUrl) return;
    let three, renderer, scene, camera, mesh, controls, raf;
    const init = async () => {
      try {
        three = await import("three");
        const OrbitControlsModule = await import("three/examples/jsm/controls/OrbitControls.js");
        const OrbitControls = OrbitControlsModule.OrbitControls || OrbitControlsModule.default;

        renderer = new three.WebGLRenderer({ antialias: true, alpha: false });
        renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        renderer.setSize(width, height);
        renderer.domElement.style.width = "100%";
        renderer.domElement.style.height = "100%";
        renderer.domElement.style.display = "block";
        renderer.domElement.style.cursor = "grab";

        scene = new three.Scene();
        camera = new three.PerspectiveCamera(50, width / height, 0.1, 100);
        camera.position.set(0, 0, 2.5);

        const light = new three.HemisphereLight(0xffffff, isDarkMode ? 0x1f2937 : 0xe2e8f0, 1);
        const dirLight = new three.DirectionalLight(0xffffff, 0.5);
        dirLight.position.set(5, 5, 5);
        scene.add(dirLight);
        scene.add(light);

        const loader = new three.TextureLoader();
        loader.setCrossOrigin("Anonymous");
        const texture = await new Promise((res, rej) => loader.load(imageUrl, res, undefined, rej));
        texture.colorSpace = three.SRGBColorSpace;
        texture.anisotropy = 8;

        const planeW = 2.0; // normalized width
        const planeH = planeW * (texture.image?.height || height) / (texture.image?.width || width);
        const geo = new three.PlaneGeometry(planeW, planeH, 32, 32);
        const mat = new three.MeshStandardMaterial({ map: texture, roughness: 0.8, metalness: 0.05 });
        mesh = new three.Mesh(geo, mat);
        scene.add(mesh);

        // OrbitControls for drag/rotate/zoom
        controls = new OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;
        controls.dampingFactor = 0.05;
        controls.minDistance = 1.5;
        controls.maxDistance = 5;
        controls.enablePan = false;
        controls.autoRotate = false;

        renderer.domElement.addEventListener("mousedown", () => {
          renderer.domElement.style.cursor = "grabbing";
        });
        renderer.domElement.addEventListener("mouseup", () => {
          renderer.domElement.style.cursor = "grab";
        });

        const animate = () => {
          if (stateRef.current.disposed) return;
          controls.update();
          renderer.render(scene, camera);
          raf = requestAnimationFrame(animate);
        };

        if (containerRef.current) {
          containerRef.current.innerHTML = "";
          containerRef.current.appendChild(renderer.domElement);
          containerRef.current.style.minHeight = `${height}px`;
        }
        animate();
      } catch (err) {
        console.error("ThreeShowcase error:", err);
        if (containerRef.current) {
          containerRef.current.innerHTML = `<div style="color:#ef4444;text-align:center;padding:20px">3D preview error: ${err.message}</div>`;
        }
      }
    };
    init();

    return () => {
      stateRef.current.disposed = true;
      if (raf) cancelAnimationFrame(raf);
      try {
        controls?.dispose?.();
        renderer?.dispose();
      } catch (_) {}
    };
  }, [imageUrl, width, height, isDarkMode]);

  return (
    <div ref={containerRef} style={{ width, height, margin: "0 auto", background: isDarkMode ? "#1f1f1f" : "#f9fafb", borderRadius: "16px", overflow: "hidden" }} />
  );
};

export default ThreeShowcase;


