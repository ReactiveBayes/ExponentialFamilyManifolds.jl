using Plots

"""
Helper function to visualize and compare two different approximations
to the target distribution.
"""
function plot_comparison(q1, q2, target_dist, title_q1, title_q2)
    x = range(-8, 8; length=100)
    y = range(-8, 8; length=100)

    z_target = [pdf(target_dist, [xi, yi]) for yi in y, xi in x]
    z_p = [pdf(q1, [xi, yi]) for yi in y, xi in x]
    z_q = [pdf(q2, [xi, yi]) for yi in y, xi in x]

    plt1 = contour(
        x,
        y,
        z_target;
        fill=true,
        alpha=0.4,
        color=:viridis,
        levels=15,
        title=title_q1,
        xlabel="x1",
        ylabel="x2",
    )
    contour!(plt1, x, y, z_p; color=:red, fill=false, linewidth=2, levels=10)

    plt2 = contour(
        x,
        y,
        z_target;
        fill=true,
        alpha=0.4,
        color=:viridis,
        levels=15,
        title=title_q2,
        xlabel="x1",
        ylabel="x2",
    )
    contour!(plt2, x, y, z_q; color=:blue, fill=false, linewidth=2, levels=10)

    return plot(plt1, plt2; layout=(1, 2), size=(1200, 400))
end
