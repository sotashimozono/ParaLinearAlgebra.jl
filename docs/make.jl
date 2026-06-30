using ParaLinearAlgebra
using Documenter
using Downloads

assets_dir = joinpath(@__DIR__, "src", "assets")
mkpath(assets_dir)
favicon_path = joinpath(assets_dir, "favicon.ico")
logo_path = joinpath(assets_dir, "logo.png")

Downloads.download("https://github.com/sotashimozono.png", favicon_path)
Downloads.download("https://github.com/sotashimozono.png", logo_path)

makedocs(;
    sitename="ParaLinearAlgebra.jl",
    format=Documenter.HTML(;
        canonical="https://codes.sota-shimozono.com/ParaLinearAlgebra.jl/stable/",
        prettyurls=get(ENV, "CI", "false") == "true",
        mathengine=MathJax3(
            Dict(
                :tex => Dict(
                    :inlineMath => [["\$", "\$"], ["\\(", "\\)"]],
                    :tags => "ams",
                    :packages => ["base", "ams", "autoload", "physics"],
                ),
            ),
        ),
        assets=["assets/favicon.ico", "assets/custom.css"],
    ),
    modules=[ParaLinearAlgebra],
    # doc-completeness is a follow-up (const aliases ⊗/paraconj, full API page);
    # keep these as warnings so the build is not blocked. Tracked as an issue.
    warnonly=[:cross_references, :missing_docs],
    pages=["Home" => "index.md"],
)

deploydocs(;
    versions=["stable", "dev"],
    repo="github.com/sotashimozono/ParaLinearAlgebra.jl.git",
    devbranch="main",
    push_preview=true,
)
