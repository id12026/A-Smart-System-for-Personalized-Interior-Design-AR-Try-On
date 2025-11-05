import { useEffect, useRef } from "react";

// Minimal WebXR AR using three.js. Places boxes on tap with a reticle.
// Works on Chrome with WebXR + AR supported hardware.

const ARWebXR = () => {
  const containerRef = useRef(null);
  const stateRef = useRef({ running: false });

  useEffect(() => {
    let renderer, scene, camera, controller, reticle, xrSession;
    let three; // dynamic import to avoid heavy initial bundle

    const init = async () => {
      try {
        three = await import("three");
        const { ARButton } = await import("three/examples/jsm/webxr/ARButton.js");

        renderer = new three.WebGLRenderer({ antialias: true, alpha: true });
        renderer.setSize(680, 460);
        renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        renderer.xr.enabled = true;
        renderer.domElement.style.borderRadius = "12px";
        renderer.domElement.style.boxShadow = "0 8px 30px rgba(0,0,0,0.2)";

        scene = new three.Scene();
        camera = new three.PerspectiveCamera(70, 680 / 460, 0.01, 20);

        const light = new three.HemisphereLight(0xffffff, 0xbbbbff, 1);
        scene.add(light);

        // Reticle for hit-test placement
        reticle = new three.Mesh(
          new three.RingGeometry(0.08, 0.1, 32).rotateX(-Math.PI / 2),
          new three.MeshBasicMaterial({ color: 0x0ea5e9 })
        );
        reticle.matrixAutoUpdate = false;
        reticle.visible = false;
        scene.add(reticle);

        controller = renderer.xr.getController(0);
        controller.addEventListener("select", () => {
          if (!reticle.visible) return;
          const geo = new three.BoxGeometry(0.15, 0.15, 0.15);
          const mat = new three.MeshStandardMaterial({ color: 0x22c55e, roughness: 0.4, metalness: 0.1 });
          const mesh = new three.Mesh(geo, mat);
          mesh.position.setFromMatrixPosition(reticle.matrix);
          scene.add(mesh);
        });
        scene.add(controller);

        const button = ARButton.createButton(renderer, { requiredFeatures: ["hit-test"] });
        button.style.marginTop = "12px";
        if (containerRef.current) {
          containerRef.current.innerHTML = "";
          containerRef.current.appendChild(renderer.domElement);
          containerRef.current.appendChild(button);
        }

        // Hit test setup
        let hitTestSource = null;
        let localSpace = null;
        renderer.xr.addEventListener("sessionstart", async () => {
          xrSession = renderer.xr.getSession();
          const viewerSpace = await xrSession.requestReferenceSpace("viewer");
          hitTestSource = await xrSession.requestHitTestSource({ space: viewerSpace });
          localSpace = await xrSession.requestReferenceSpace("local");
        });
        renderer.xr.addEventListener("sessionend", () => {
          hitTestSource = null;
          xrSession = null;
        });

        const clock = new three.Clock();
        const renderLoop = (timestamp, frame) => {
          const delta = clock.getDelta();
          if (frame && xrSession && hitTestSource && localSpace) {
            const hitTestResults = frame.getHitTestResults(hitTestSource);
            if (hitTestResults.length) {
              const pose = hitTestResults[0].getPose(localSpace);
              reticle.visible = true;
              reticle.matrix.fromArray(pose.transform.matrix);
            } else {
              reticle.visible = false;
            }
          }
          renderer.render(scene, camera);
        };
        renderer.setAnimationLoop(renderLoop);

        stateRef.current.running = true;
      } catch (err) {
        console.error("WebXR init error:", err);
        if (containerRef.current) {
          containerRef.current.innerHTML = "<div style=\"color:#ef4444;text-align:center\">WebXR not supported on this device/browser.</div>";
        }
      }
    };

    init();

    return () => {
      try {
        stateRef.current.running = false;
        if (renderer) {
          renderer.setAnimationLoop(null);
          const session = renderer.xr.getSession?.();
          session?.end?.();
          renderer.dispose?.();
        }
      } catch (_) {}
    };
  }, []);

  return (
    <div ref={containerRef} style={{ width: "100%", display: "flex", flexDirection: "column", alignItems: "center" }} />
  );
};

export default ARWebXR;


