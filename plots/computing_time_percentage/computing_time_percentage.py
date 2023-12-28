# Import libraries
import matplotlib.pyplot as plt
import matplotlib.patches as patches


# Config
color_scheme = ["#366CC7".lower(), "#70C4A5".lower(), "#2B2191".lower(), "#FDF2BE".lower()]
# Create figure() objects
figsize_cfg = (5, 5)
fig = plt.figure(figsize=figsize_cfg)

# Creating two axes
ax=fig.add_axes([0,0,1,1])
xlim_cfg = 5.2
ylim_cfg = 5.2
ax.set_ylim(0, xlim_cfg)
ax.set_xlim(0, ylim_cfg)
ax.set_prop_cycle(color=color_scheme)
# Prepare data
data = [0.5763, 0.8944, 1.3e-03]
data_text = ["  0.57", "  0.89", "1.3e-03"]
labels = ["Generation", "Ranking", "Unlinkable prediction"]
prop = [i/sum(data) for i in data]
t = [i*figsize_cfg[0] for i in prop]
# Draw stacked bar
w = 0.8
b1 = ax.barh(y=2.5, width=t[0], height=w)
b2 = ax.barh(y=2.5, width=t[1], left=t[0], height=w)
b3 = ax.barh(y=2.5, width=t[2], left=t[0]+t[1], height=w)
ax.legend([b1, b2, b3], labels,
          bbox_to_anchor=(0.45, 0.82),
          loc="upper left", prop={'size': 10})

# Add text under plot bar
for i, p, d in zip(b1+b2, prop, data_text):
    h = i.get_height()
    ax.text(i.get_x()-0.2+ i.get_width()/2, h+2.2, "{}s \n ({:.2f}%)".format(d, p*100))

ax.text(3.6, 0.4, "{}s \n ({:.2f}%)".format(data_text[2], prop[2]*100))

# Draw a rectangle
rect_config_x = 4.99
rect_config_y = 2.49
rect_config_width = 0.01
rect_config_height = 0.01
rect = patches.Rectangle(xy=(rect_config_x, rect_config_y),
                         width=rect_config_width,
                         height=rect_config_height,
                         fill=None,
                         edgecolor='#000000')
rect_ul = (rect_config_x, rect_config_y + rect_config_height)
rect_dr = (rect_config_x + rect_config_width, rect_config_y)
ax.add_patch(rect)

axes_ins_cfg = [0.6, 0.15, 0.16, 0.16]
axes_ins=fig.add_axes(axes_ins_cfg)
axes_ins.set_prop_cycle(color=color_scheme[1:])
axes_ins.tick_params(top=False, bottom=False, left=False, right=False, labelbottom=False, labelleft=False)

# Compute top left and bot right coor
axes_ul = (axes_ins_cfg[0]*xlim_cfg, (axes_ins_cfg[1]+axes_ins_cfg[3])*ylim_cfg)
axes_dr = ((axes_ins_cfg[0] + axes_ins_cfg[2])*xlim_cfg, axes_ins_cfg[1]*ylim_cfg)

# Draw second axes
axes_ins_cfg_xlim = 1
axes_ins_cfg_ylim = 1
axes_ins.set_ylim([0, axes_ins_cfg_ylim])
axes_ins.set_xlim([0, axes_ins_cfg_xlim])
r = figsize_cfg[0]/rect_config_height
snd_data = [sum(data)/r - data[2], data[2]]
prop = [i/sum(snd_data) for i in snd_data]
t = [i*axes_ins_cfg_xlim for i in prop]
b1 = axes_ins.barh(y=0.5, width=t[0], height=axes_ins_cfg_ylim)
b2 = axes_ins.barh(y=0.5, width=t[1], left=t[0], height=axes_ins_cfg_ylim)

# Draw line
ax.plot((rect_ul[0], axes_ul[0]), (rect_ul[1], axes_ul[1]), linewidth=0.3, color="#000000")
ax.plot((rect_dr[0], axes_dr[0]), (rect_dr[1], axes_dr[1]), linewidth=0.3, color="#000000")

# Remove border
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

# Show plot
plt.savefig("ExecTimeAttribution.pdf")
plt.show()
