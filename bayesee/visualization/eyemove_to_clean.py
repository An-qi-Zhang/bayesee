#%%
import cv2
import numpy as np
import glob
import matplotlib as mpl
from matplotlib.pyplot import *

#%%
rcParams['figure.figsize'] = [8,6]
rcParams['figure.dpi'] = 200
rcParams['savefig.format'] = 'pdf'
rcParams['font.size'] = '12'

def coordinate_mat_to_cart(x, y, l):
    return y, x

#%%
def print_frames(p_fix, loc_fix, p_map, p_map0, loc_target=None, path_save='.'):
    fix_0_max = sum(p_fix>0)
    for i in range(fix_0_max):
        fig, ax = subplots()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        im = imshow(p_map[i,:,:], cmap='YlOrBr', norm=mpl.colors.LogNorm(vmin=1e-10,vmax=1))
        fig.colorbar(im)
        ax.set_title('n_fix={}, p_to_fix={:.3g}, p_absent={:.3g}'.format(i, p_fix[i], p_map0[i]))

        if i>0:
            for j in range(1, i+1):
                x, y = coordinate_mat_to_cart(loc_fix[j-1,0], loc_fix[j-1,1], p_map.shape[2])
                plot(x, y, 'g+', markersize=20*(j+1)/(i+1), alpha=(j+1)/(i+1))
                x2, y2 = coordinate_mat_to_cart(loc_fix[j,0], loc_fix[j,1], p_map.shape[2])
                arrow(x, y, x2-x, y2-y, width=2, fc='r', ec='r', alpha=(j+1)/(i+1))
                if p_fix[i] == p_map0[i]:
                    plot(p_map.shape[1]//2, p_map.shape[2]//2, 'rX', markersize=p_map.shape[2]//2, alpha=0.3*(j==i))
        
        if loc_target is not None:
            plot(loc_target[1], loc_target[0], 'b*', markersize=18, alpha=0.3)

        savefig(path_save+'/fix_id'+str(i).zfill(2)+'.png')

def video(theme, path, fps, path_save='.'):
    images = []
    for filename in glob.glob(path + '/*.png'):
        img = cv2.imread(filename)
        images.append(img)
    height, width, layers = img.shape

    out = cv2.VideoWriter(path_save+'/'+theme+'.mp4',cv2.VideoWriter_fourcc(*'MP4V'), fps, (width,height))

    for i in range(len(images)):
        out.write(images[i])

    out.release()

# %%
def perf_portfolio(theme, max_fix, n_fix, dis, acc, f_loc_fix, ps_loc_fix, p_n2_err_dfix, search_l, fix_cost, t_mask, path_save='.'):
    fig = figure(figsize=(12, 5))
    gs = fig.add_gridspec(5,18)
    ax0 = fig.add_subplot(gs[1:4,0:4])
    ax0.ticklabel_format(style='sci', axis='y', scilimits=(-2,2))
    ax0.hist(n_fix-1, bins=range(max_fix+1), rwidth=0.9, align='right')
    ax0.set_xlabel('number of fixation')
    ax0.set_ylabel('frequency')

    ax1 = fig.add_subplot(gs[1:4,5:9])
    ax1.ticklabel_format(style='sci', axis='y', scilimits=(-2,2))
    ax1.hist(dis[0], bins=20, rwidth=0.9, edgecolor='None', alpha = 0.2, color= 'gray', label='all')
    ax1.hist(dis[1], bins=20, rwidth=0.9, edgecolor='None', alpha = 0.6, color= 'yellow', label='present')
    ax1.hist(dis[2], bins=20, rwidth=0.9, edgecolor='None', alpha = 0.2, color= 'green', label='absent')
    ax1.set_xlabel('saccade amplitude (pixels)')
    ax1.legend(prop={'size': 10})

    ax2 = fig.add_subplot(gs[1:4,10:14])
    ax2.ticklabel_format(style='sci', axis='y', scilimits=(-2,2))
    ax2.hist(acc, bins=range(-2,4,1), rwidth=0.9,align='left', weights=np.ones(len(acc)) / len(acc))
    xticks([-2,-1,0,1,2], ['false hit', 'false alarm', 'miss', 'correct reject', 'hit'], fontsize=8, rotation=30)
    ax2.set_ylim([0,0.5])

    ax3 = fig.add_subplot(gs[1:4,14:18])
    im = ax3.imshow(f_loc_fix*t_mask, cmap='CMRmap')
    ax3.contour(t_mask, colors='white', linewidths=0.1, linestyles='dotted')
    ax3.get_xaxis().set_visible(False)
    ax3.get_yaxis().set_visible(False)
    ax3.axes.set_title("fixation heat map({}x{})".format(f_loc_fix.shape[0],f_loc_fix.shape[1]), fontsize=10)
    fig.colorbar(im, cax=axes([0.925, 0.3, 0.01, 0.4]), label='Relative Freq')

    # ax4 = fig.add_subplot(gs[3:5,0:2])
    # ax4.imshow(p_n2_err_dfix, cmap='CMRmap')
    # ax4.contour(t_mask, colors='white', linewidths=0.1, linestyles='dotted')
    # ax4.get_xaxis().set_visible(False)
    # ax4.get_yaxis().set_visible(False)
    # ax4.axes.set_title("false hit difference ({}x{})".format(p_n2_err_dfix.shape[0],p_n2_err_dfix.shape[1]), fontsize=10)
    # ax4.plot(p_n2_err_dfix.shape[0]//2, p_n2_err_dfix.shape[1]//2, 'b.', markersize=3)

    # ax5 = fig.add_subplot(gs[3:5,2:4])
    # ax5.imshow(ps_loc_fix[1], cmap='CMRmap')
    # ax5.contour(t_mask, colors='white', linewidths=0.1, linestyles='dotted')
    # ax5.get_xaxis().set_visible(False)
    # ax5.get_yaxis().set_visible(False)
    # ax5.axes.set_title("false alarm fixations({}x{})".format(f_loc_fix.shape[0],f_loc_fix.shape[1]), fontsize=10)

    # ax6 = fig.add_subplot(gs[3:5,4:6])
    # ax6.imshow(ps_loc_fix[2], cmap='CMRmap')
    # ax6.contour(t_mask, colors='white', linewidths=0.1, linestyles='dotted')
    # ax6.get_xaxis().set_visible(False)
    # ax6.get_yaxis().set_visible(False)
    # ax6.axes.set_title("miss ({}x{})".format(f_loc_fix.shape[0],f_loc_fix.shape[1]), fontsize=10)

    # ax7 = fig.add_subplot(gs[3:5,6:8])
    # ax7.imshow(ps_loc_fix[3])
    # ax7.contour(t_mask, colors='white', linewidths=0.1, linestyles='dotted')
    # ax7.get_xaxis().set_visible(False)
    # ax7.get_yaxis().set_visible(False)
    # ax7.axes.set_title("correct reject ({}x{})".format(f_loc_fix.shape[0],f_loc_fix.shape[1]), fontsize=10)

    # fig.colorbar(im, cax=axes([0.1, 0.5, 0.8, 0.02]), orientation='horizontal', label="probability")

    gcf().text(0.5, 0.92, "ACCURACY - overall: {:.3g}; when present: {:.3g}; when absent: {:.3g}\nSEARCH LENGTH - overall: {:.1f}; when present {:.1f}; when absent {:.1f}\nFIXATION COST - overall: {:.1f}; when present {:.1f}; when absent {:.1f}".format((acc>0).sum()/len(acc), (acc==2).sum()/(acc%2==0).sum(), (acc==1).sum()/(acc%2==1).sum(), search_l[0], search_l[1], search_l[2], fix_cost[0], fix_cost[1], fix_cost[2]), fontsize=8, ha="center", bbox={"edgecolor":"red", "facecolor":"white", "pad":5})

    savefig(path_save+'/'+theme+'.pdf')
    close()

# %%
def vars2d_surface(theme, vars1, vars2, f, vars1name='var1', vars2name='var2', angles=((45,90),(45,0)), path_save='.'):
    fig = figure(figsize=(10,8))
    fig.tight_layout()
    gs = fig.add_gridspec(1, 2)
    ax0 = fig.add_subplot(gs[0,0], projection='3d')
    ax1 = fig.add_subplot(gs[0,1], projection='3d')
    cax = axes([0.1, 0.8, 0.8, 0.05])

    ax0.view_init(angles[0][0], angles[0][1])
    var1m, var2m = np.meshgrid(vars1, vars2, indexing='ij')
    p = ax0.plot_surface(var1m, var2m, f, edgecolor='white', cmap='plasma')
    ax0.set_xlabel(vars1name, fontsize=12)
    ax0.set_ylabel(vars2name, fontsize=12)

    ax1.view_init(angles[1][0], angles[1][1])
    p = ax1.plot_surface(var1m, var2m, f, edgecolor='white', cmap='plasma')
    ax1.set_xlabel(vars1name, fontsize=12)
    ax1.set_ylabel(vars2name, fontsize=12)

    fig.colorbar(p, cax=cax, orientation='horizontal', label=theme)

    savefig(path_save+'/'+theme+'.pdf')

# %%
def vars2d_lines(theme, vars1, vars2, f, vars1name='var1', vars2name='var2', path_save='.'):
    fig = figure()
    fig.tight_layout()
    for i in range(len(vars1)):
        plot(vars2, f[i,:], '.-', label=vars1name+':{:.2g}'.format(vars1[i]))

    xlabel(vars2name, fontsize=12)
    ylabel(theme, fontsize=12)
    legend(loc='best')

    savefig(path_save+'/'+theme+'.pdf')
