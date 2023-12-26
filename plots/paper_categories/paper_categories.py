# %%
import matplotlib.pyplot as plt

# %%
# Plot config
color_scheme = ["#F7A482".lower(), "#F1DD88".lower(), "#F6EDC8".lower(), "#83BB90".lower(), "#549896".lower(), "#017293".lower(), "#AB3131".lower()]
plt.rcParams["font.family"] = "DeJavu Serif"
plt.rcParams["font.serif"] = ["Times New Roman"]
# Create figure objects
figsize_cfg = (5, 5)
fig = plt.figure(figsize=figsize_cfg)

# %%
# Creating two axes
ax = fig.add_axes([0, 0, 1, 1])
xlim_cfg = 5
ylim_cfg = 8
ax.set_xlim(0, xlim_cfg)
ax.set_ylim(0, ylim_cfg)
ax.axis("off")
ax.set_prop_cycle(color=color_scheme)
ax.tick_params(top=False, bottom=False, left=False, right=False, labelbottom=False, labelleft=False)

# %%
# Data
data = [169, 409, 195, 188, 176, 254, 229]
print(sum(data))
data_text = [str(i) for i in data]
label = ["Công nghệ", "Giải trí", "Giáo dục", "Khoa học", "Kinh tế", "Thế giới", "Văn hóa"] 

prop = [i/sum(data) for i in data]
t = [i*figsize_cfg[0]*2.5 for i in prop]

# %%
# Draw stacked bar
w = 0.6
start_anchor = 0.2
b1 = ax.barh(y=1, width=t[0], left=start_anchor, height=w)
b2 = ax.barh(y=2, width=t[1], left=start_anchor, height=w)
b3 = ax.barh(y=3, width=t[2], left=start_anchor, height=w)
b4 = ax.barh(y=4, width=t[3], left=start_anchor, height=w)
b5 = ax.barh(y=5, width=t[4], left=start_anchor, height=w)
b6 = ax.barh(y=6, width=t[5], left=start_anchor, height=w)
b7 = ax.barh(y=7, width=t[6], left=start_anchor, height=w)

# %%
# Draw legend
ax.legend([b7, b6, b5, b4, b3, b2, b1], 
          ["Văn hóa", "Thế giới", "Kinh tế", "Khoa học", "Giáo dục", "Giải trí", "Công nghệ"], 
          bbox_to_anchor=(0.64, 0.92),
          loc="upper left",
          prop={"size": 10})

# %%
# Add text
for i, p, d in zip(b1 + b2 + b3 + b4 + b5 + b6 + b7, prop, data_text):
    h = i.get_height()
    px = i.get_xy()[1]
    ax.text(i.get_x() - 0.14 + i.get_width()/2 - 0.17, px + 0.1, "   {}\n({:.2f}%)".format(d, p*100))
# %%
# Show plot
plt.savefig("paper_categories.pdf")
plt.show()
