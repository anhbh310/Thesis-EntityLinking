# %%
import matplotlib.pyplot as plt

# %%
# Plot config
color_scheme = ["#70C4A5".lower(), "#366CC7".lower(), "#2B2191".lower(), "#FDF2BE".lower()]
plt.rcParams["font.family"] = "DeJavu Serif"
plt.rcParams["font.serif"] = ["Times New Roman"]
# Create figure objects
figsize_cfg = (5, 1.5)
fig = plt.figure(figsize=figsize_cfg)

# %%
# Creating two axes
ax = fig.add_axes([0, 0, 1, 1])
xlim_cfg = 5.5
ylim_cfg = 6.0
ax.set_xlim(0, xlim_cfg)
ax.set_ylim(0, ylim_cfg)
ax.axis("off")
ax.set_prop_cycle(color=color_scheme)
ax.tick_params(top=False, bottom=False, left=False, right=False, labelbottom=False, labelleft=False)

# %%
# Data
data = [160, 1460]
data_text = [str(i) for i in data]
label = ["NIL", ""]

prop = [i/sum(data) for i in data]
print(prop)
t = [i*figsize_cfg[0] for i in prop]

# %%
# Draw stacked bar
w = 1
start_anchor = 0.2
b1 = ax.barh(y=1.9, width=t[0], left=start_anchor, height=w)
b2 = ax.barh(y=1.9, width=t[1], left=start_anchor + t[0], height=w)

# %%
# Draw legend
ax.legend([b1, b2], 
          ["Nil", "Others"], 
          bbox_to_anchor=(0.64, 0.92),
          loc="upper left",
          prop={"size": 10})

# %%
# Add text
for i, p, d in zip(b1 + b2, prop, data_text):
    h = i.get_height()
    ax.text(i.get_x() - 0.14 + i.get_width()/2 - 0.12, h - 0.9, "  {}\n({:.2f}%)".format(d, p*100))
# %%
# Show plot
plt.savefig("nil_entities.pdf")
plt.show()
