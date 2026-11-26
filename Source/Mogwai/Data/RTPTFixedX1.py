import falcor


def render_graph_rtpt_fixed_x1(denoise=True, path_options=None):
    """Render fixed-x1 ReSTIR PT, exposing the raw and reconstructed outputs."""
    graph = falcor.RenderGraph("RTPTFixedX1")
    for library in ("GBuffer", "ScreenSpaceReSTIRPass", "RTPT", "ModulateIllumination"):
        falcor.loadRenderPassLibrary(library + ".dll")

    graph.addPass(falcor.createPass("GBufferRT", {
        "samplePattern": falcor.SamplePattern.Center,
        "sampleCount": 1,
        "texLOD": falcor.TexLODMode.Mip0,
    }), "GBuffer")
    graph.addPass(falcor.createPass("ScreenSpaceReSTIRPass"), "Direct")
    options = {
        "samplesPerPixel": 1,
        "candidateSamples": 1,
        "maxSurfaceBounces": 5,
        "enableTemporalReuse": True,
        "enableSpatialReuse": True,
        "useDirectLighting": True,
        "enablePG": True,  # Share the diffuse guide across Generate, Talbot Temporal, and Pairwise Spatial.
        "useNRDDemodulation": True,
    }
    options.update(path_options or {})
    options.setdefault("enableSHARC", options["enablePG"])
    graph.addPass(falcor.createPass("RTPT", options), "PT")

    for target in ("Direct", "PT"):
        graph.addEdge("GBuffer.vbuffer", target + ".vbuffer")
        graph.addEdge("GBuffer.mvec", target + ".motionVectors")
    graph.addEdge("Direct.color", "PT.directLighting")
    graph.addEdge("Direct.emission", "PT.directEmission")
    for signal in ("directDiffuseRadianceHitDist", "directSpecularRadianceHitDist"):
        graph.addEdge("Direct." + signal, "PT." + signal)

    def add_composite(name, diffuse, specular):
        graph.addPass(falcor.createPass("ModulateIllumination"), name)
        graph.addEdge(diffuse, name + ".diffuseRadiance")
        graph.addEdge(specular, name + ".specularRadiance")
        graph.addEdge("PT.nrdDiffuseReflectance", name + ".diffuseReflectance")
        graph.addEdge("PT.nrdSpecularReflectance", name + ".specularReflectance")
        graph.addEdge("PT.nrdResidualRadianceHitDist", name + ".residualRadiance")
        graph.addEdge("PT.nrdEmission", name + ".emission")
        graph.markOutput(name + ".output")

    if denoise:
        falcor.loadRenderPassLibrary("NRDPass.dll")
        nrd = falcor.createPass("NRD", {"worldSpaceMotion": True})
        if "worldSpaceMotion" not in nrd.getDictionary():
            raise RuntimeError("RTPTFixedX1 needs the RELAX-enabled NRDPass for denoising.")
        graph.addPass(nrd, "NRD")
        graph.addEdge("GBuffer.mvecW", "NRD.mvec")
        graph.addEdge("GBuffer.linearZ", "NRD.viewZ")
        graph.addEdge("GBuffer.normWRoughnessMaterialID", "NRD.normWRoughnessMaterialID")
        graph.addEdge("PT.nrdDiffuseRadianceHitDist", "NRD.diffuseRadianceHitDist")
        graph.addEdge("PT.nrdSpecularRadianceHitDist", "NRD.specularRadianceHitDist")
        add_composite("FilteredComposite", "NRD.filteredDiffuseRadianceHitDist", "NRD.filteredSpecularRadianceHitDist")

    add_composite("Composite", "PT.nrdDiffuseRadianceHitDist", "PT.nrdSpecularRadianceHitDist")
    graph.markOutput("PT.color")
    return graph


if "m" in globals() and globals().get("__name__", "__main__") == "__main__":
    m.addGraph(render_graph_rtpt_fixed_x1())
    m.loadScene("VeachAjar/VeachAjar.pyscene")
