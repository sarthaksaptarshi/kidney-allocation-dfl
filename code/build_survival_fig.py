import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

ci = mpimg.imread("/mnt/user-data/uploads/dafb9d46-image.png")   # C-index bars (death-censored)
ca = mpimg.imread("/mnt/user-data/uploads/0e0d4bc3-image.png")   # calibration (death-censored)

fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.1),
                         gridspec_kw={"width_ratios": [ci.shape[1]/ci.shape[0],
                                                       ca.shape[1]/ca.shape[0]]})
for ax, im, lab in zip(axes, [ci, ca], ["(a)", "(b)"]):
    ax.imshow(im); ax.axis("off")
    ax.text(0.01, 0.99, lab, transform=ax.transAxes, fontsize=13,
            fontweight="bold", va="top", ha="left")
fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0.02)
fig.savefig("out/fig_survival.png", dpi=170, bbox_inches="tight", pad_inches=0.02)
print("wrote out/fig_survival.png")
