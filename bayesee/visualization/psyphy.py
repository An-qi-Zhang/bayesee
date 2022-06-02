
    #%%        
def exp_th(targets, backgrounds, sessions, path='.'):

    db_exp_th, db_CI68 = load_ths(path, sessions)

    fig, ax = subplots(figsize=(15,8))
    errorbar(targets, db_exp_th[:len(targets)], yerr=db_CI68[:,:len(targets)],
        fmt='ko', label=backgrounds[0], mew=3, mfc='w', capsize=30, markersize=20)
    errorbar(targets, db_exp_th[len(targets):], yerr=db_CI68[:,len(targets):],
        fmt='ko', label=backgrounds[1], mew=3, capsize=30, markersize=20)
    legend(loc='lower left', fontsize=36)
    ylabel('Amplitude threshold(dB)', fontsize=36)

    # exp1
    xlim([-0.5, 4.5])
    ylim([32, 64]) 
    ax.set_xticks([0,1,2,3,4])
    ax.set_xticks([0.5,1.5,2.5,3.5], minor=True)
    ax.set_yticks([34,38,42,46,50,54,58,62])
    ax.set_yticks([32,36,40,44,48,52,56,60,64], minor=True)

    # exp2
    # xlim([-0.5, 3.5])
    # ylim([28, 48])
    # ax.set_xticks([0,1,2,3])
    # ax.set_xticks([0.5,1.5,2.5], minor=True)
    # ax.set_yticks([30,34,38,42,46])
    # ax.set_yticks([28,32,36,40,44,48], minor=True)

    ax.tick_params(axis='x', which='both', direction='out', length=0, width=0,pad = 5, labelsize=36, labelbottom=True, labeltop=False, grid_color='k', grid_alpha=1, grid_linewidth=1, grid_linestyle='--')
    ax.grid(b=True, which='minor', axis='x', linestyle='--', linewidth=2)
    ax.tick_params(axis='y', which='major', direction='out', length=12, width=4, pad =3, labelsize=36, left=True, right=True, labelleft=True, labelright=True)
    ax.tick_params(axis='y', which='minor', direction='out', length=8, width=4, left=True, right=True, labelleft=False, labelright=False)

    savefig(path+'/exp_th'+'{}.pdf'.format(sessions))
    savefig(path+'/exp_th'+'{}.jpeg'.format(sessions))
    close()

#%%
def cmp_th(targets, backgrounds, models, cmp_th, path='.', theme=''):

    nmods = len(models)
    nrows = nmods//2
    ncols = nmods//2

    db_cmp_th = np.zeros((nmods,len(backgrounds),len(targets)))
    db_cmp_th[:,0,:] = decibel(cmp_th[nmods:,:])
    db_cmp_th[:,1,:] = decibel(cmp_th[:nmods,:])

    fig = figure(figsize=(32, 16))
    gs = fig.add_gridspec(nrows, ncols)
    colors = ['steelblue', 'firebrick', 'forestgreen', 'indigo']

    for i in range(nrows):
        for j in range(ncols):
            ax = fig.add_subplot(gs[i,j])
            
            ax.scatter(targets, db_cmp_th[i*nrows+j,0,:], s=500, marker='D', edgecolor=colors[i*nrows+j],  facecolor="none", linewidths=3,  zorder=1)
            ax.scatter(targets, db_cmp_th[i*nrows+j,1,:], s=500, marker='D', color=colors[i*nrows+j], label=models[i*nrows+j], zorder=1)

            ax.set_ylim(db_cmp_th.flatten().min()-5, db_cmp_th.flatten().max()+5)
            ax.tick_params(axis='x', which='both', direction='out', length=0, width=0,pad =5, labelsize=36, labelbottom=True, labeltop=False, grid_color='k', grid_alpha=1, grid_linewidth=1, grid_linestyle='--')
            ax.grid(b=True, which='minor', axis='x', linestyle='--', linewidth=2)
            ax.tick_params(axis='y', which='major', direction='out', length=12, width=4, pad = 3, labelsize=36, left=True, right=True, labelleft=True, labelright=True)
            ax.tick_params(axis='y', which='minor', direction='out', length=8, width=4, left=True, right=True, labelleft=False, labelright=False)

            if j == 1:
                ax.tick_params(axis='y', labelleft=False)
            if i == 0:
                ax.tick_params(axis='x', labelbottom=False)

            legend(loc='lower left', fontsize=36)

    gcf().text(0.002, 0.5, 'Adjusted amplitude threshold(dB)', va='center', rotation='vertical',fontsize=48)
    tight_layout(pad=2.5, h_pad=0.5, w_pad=0.5)

    savefig(path+'/{}cmp_th.pdf'.format(theme))
    savefig(path+'/{}cmp_th.png'.format(theme))
    close()

#%%
def psychometric(amps, acc, subject, path_save='.'):

    exp_slopes = np.load(path_save+'/bs_slopes_{}.npy'.format(subject))
    exp_shapes = np.load(path_save+'/bs_shapes_{}.npy'.format(subject))
    exp_criteria = np.load(path_save+'/bs_criteria_{}.npy'.format(subject))
    exp_ths = np.load(path_save+'/bs_ths_{}.npy'.format(subject))

    exp_slope = np.mean(exp_slopes,axis=0)
    exp_shape = np.mean(exp_shapes,axis=0)
    exp_criterion = np.mean(exp_criteria,axis=0)
    exp_th = np.mean(exp_ths,axis=0)
    
    p_acc = acc.mean(axis=0) # percent accuracy
    nrows= 1
    ncols= np.size(p_acc, 1) #nConditions
    
    fig = figure(figsize=(15,8))
    gs = fig.add_gridspec(nrows,ncols)

    amp_min = amps.min() * 0.8
    amp_max = amps.max() * 1.1
    amp_fit = np.linspace(amp_min, amp_max, 500)

    for i in range(ncols):
            ax = fig.add_subplot(gs[0,i])

            for j in range(amps.shape[0]):
                ax.scatter(amps[j,i,:], p_acc[j,i,:], color='k')

            ax.plot(amp_fit, (norm.cdf(0.5*(amp_fit/exp_slope[i])**exp_shape[i]-exp_criterion[i])+1-norm.cdf(-0.5*(amp_fit/exp_slope[i])**exp_shape[i]-exp_criterion[i]))/2.0, color='b')
            axhline(0.69, color='g', alpha=0.5)
            axvline(exp_th[i], color='r', alpha=0.5)

            text(amp_min+(amp_max-amp_min)*0.1, 1.05,r'$\alpha$={:.3f}'.format(exp_slope[i]),  fontsize=16)
            text(amp_min+(amp_max-amp_min)*0.1, 1,r'$\beta$={:.3f}'.format(exp_shape[i]), fontsize=16)
            text(amp_min+(amp_max-amp_min)*0.1, 0.95,r'$\gamma$={:.3f}'.format(exp_criterion[i]), fontsize=16)

            xlim([amp_min, amp_max])
            ylim([0.4, 1.1])
            xticks(np.linspace(amp_min, amp_max, 5), fontsize=16, rotation=30)
            yticks(np.linspace(0.4, 1, 7))

    gcf().text(0.5, 0.05, 'Target Amplitude', ha='center', fontsize=36)
    gcf().text(0.01, 0.5, 'Percentage Correct', va='center', rotation='vertical', fontsize=36)
    gcf().text(0.3, 0.9, 'Low Amplitude Similarity', ha='center', fontsize=24)
    gcf().text(0.7, 0.9, 'High Amplitude Similarity', ha='center', fontsize=24)

    savefig(path_save+'/psychometric_{}.pdf'.format(subject))
    savefig(path_save+'/psychometric_{}.jpeg'.format(subject))
    close()

#%%
def bin_psychometric(bin_amps, bin_acc, subject, path_save='.'):

    nrows = bin_amps.shape[2]
    ncols = bin_amps.shape[0]
    fig = figure(figsize=(35,8))
    gs = fig.add_gridspec(nrows,ncols)
    
    amp_min = bin_amps.min() * 0.8
    amp_max = bin_amps.max() * 1.1
    amp_fit = np.linspace(amp_min, amp_max, 500)
    hist_nbins = 5

    for i in range(ncols):
        exp_slopes = np.load(path_save+'/bs_slopes_{}_{}of{}.npy'.format(subject, i+1, ncols))
        exp_shapes = np.load(path_save+'/bs_shapes_{}_{}of{}.npy'.format(subject, i+1, ncols))
        exp_criteria = np.load(path_save+'/bs_criteria_{}_{}of{}.npy'.format(subject, i+1, ncols))
        exp_ths = np.load(path_save+'/bs_ths_{}_{}of{}.npy'.format(subject, i+1, ncols))

        exp_slope = np.mean(exp_slopes,axis=0)
        exp_shape = np.mean(exp_shapes,axis=0)
        exp_criterion = np.mean(exp_criteria,axis=0)
        exp_th = np.mean(exp_ths,axis=0)

        for j in range(nrows):
            ax = fig.add_subplot(gs[j,i])

            hist_amp_min = bin_amps[i,:,j].min()
            hist_amp_max = bin_amps[i,:,j].max() * 1.01
            bin_edges = np.histogram_bin_edges(bin_amps[i,:,j], bins=hist_nbins, range=(hist_amp_min,hist_amp_max))
            bin_sizes = np.zeros((hist_nbins,))

            for k in range(hist_nbins):
                bin_sizes[k] = ((bin_amps[i,:,j] >= bin_edges[k]) * (bin_amps[i,:,j] < bin_edges[k+1])).sum()

            mask = bin_acc[i,:,j] == 1
            bin_weights = np.ones((len(bin_amps[i,mask,j]),))

            interested_bin_amps = bin_amps[i,mask,j]
            for k in range(len(bin_weights)):
                l = 0
                while interested_bin_amps[k] >= bin_edges[l]:
                    l += 1
                bin_weights[k] /= bin_sizes[l-1]
        
            ax.hist(bin_amps[i,mask,j], bins=bin_edges, weights=bin_weights)

            ax.plot(amp_fit, (norm.cdf(0.5*(amp_fit/exp_slope[j])**exp_shape[j]-exp_criterion[j])+1-norm.cdf(-0.5*(amp_fit/exp_slope[j])**exp_shape[j]-exp_criterion[j]))/2.0, color='b')
            axhline(0.69, color='g', alpha=0.5)
            axvline(exp_th[j], color='r', alpha=0.5)

            text(amp_min+(amp_max-amp_min)*0.1, 1.05,r'$\alpha$={:.3f}'.format(exp_slope[j]),  fontsize=16)
            text(amp_min+(amp_max-amp_min)*0.1, 1,r'$\beta$={:.3f}'.format(exp_shape[j]), fontsize=16)
            text(amp_min+(amp_max-amp_min)*0.1, 0.95,r'$\gamma$={:.3f}'.format(exp_criterion[j]), fontsize=16)

            xlim([amp_min, amp_max])
            ylim([0.4, 1.1])
            xticks(np.linspace(amp_min, amp_max, 5), fontsize=16, rotation=30)
            yticks(np.linspace(0.4, 1, 7))

    gcf().text(0.5, 0.01, 'Target Amplitude', ha='center', fontsize=36)
    gcf().text(0.07, 0.5, 'Percentage Correct', va='center', rotation='vertical', fontsize=36)
    gcf().text(0.93, 0.75, 'Low Amplitude Similarity', va='center',rotation='vertical', fontsize=16)
    gcf().text(0.93, 0.25, 'High Amplitude Similarity', va='center', rotation='vertical', fontsize=16)

    savefig(path_save+'/bin_psychometric_{}.pdf'.format(subject))
    savefig(path_save+'/bin_psychometric_{}.jpeg'.format(subject))
    close()
