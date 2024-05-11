import numpy as np
import matplotlib.pyplot as plt
from matplotlib.tri import LinearTriInterpolator, CubicTriInterpolator, UniformTriRefiner
import matplotlib.tri as tri
from matplotlib.collections import LineCollection
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.patches import Polygon
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator
import copy
from nwkpy import _constants


def plot_bands(kzvals, bands, spdist, chemical_potential=None):
    fig = plt.figure(figsize=(7, 7))
    gs = gridspec.GridSpec(1, 1, left=0.00, right=1, top=1.0, bottom=0.0, hspace=0.0)
    
    ax = fig.add_subplot(gs[0,0])
    
    # caratteri
    nk = spdist.shape[0]
    if chemical_potential is not None:
        ax.axhline(y=chemical_potential, color='black', ls='--', lw=4)
        ax.text(0.08,chemical_potential+0.003,r"$\mu$",size=15)
        
    # bande
    Ekz0 = bands[0,:]
    mi = Ekz0.min()
    ma = Ekz0.max()
    
    EL = spdist[:,:2,:].sum(axis=1)
    HH = spdist[:,2:4,:].sum(axis=1)
    LH = spdist[:,4:6,:].sum(axis=1)
    SO = spdist[:,6:8,:].sum(axis=1)
    color = (HH+LH+SO)-EL
    # struttura a bande con colori per carattere
    ax.set_xlim(0.0, 0.1)
    ax.set_ylim(mi-0.1,ma+0.1)
    ax.set_ylabel('$E$ [eV]', size = 20)
    for j in range(0,spdist.shape[2],1):
            
        x = kzvals
        y = bands[:,j]
        
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1) 
        norm5 = plt.Normalize(-1., 1.0)
        lc = LineCollection(segments, cmap= 'jet', norm=norm5)
        lc.set_array(color[:,j])
        lc.set_linewidth(4)
        line5 = ax.add_collection(lc)
        
    cbaxes = inset_axes(ax, width="40%", height="6%", loc=1) 
    
    cbar = fig.colorbar(line5,
                 cax=cbaxes,
                 ticks=[-1,0.9],
                 orientation='horizontal')
    
    cbar.ax.set_xticklabels(['EL', 'H'],size=15)
    cbar.ax.tick_params(size=0)
    
    ax.set_xlabel('$ k_{z} ( \\frac{\pi}{\sqrt{3} a_{lc}} )$',size=20)
    
    return fig

def plot_bands(
        kzvals, 
        bands, 
        spinor_distribution, 
        norm_sum_region=None, 
        angular_momentum = None, 
        threshold_el=None, 
        threshold_h=None, 
        chemical_potential=None, 
        character_to_show=None, 
        figsize=(5, 5), 
        xlim=None, 
        ylim=None, 
        cmap_in='Blues'
    ):

    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(1, 1, left=0.00, right=1, top=1.0, bottom=0.0, hspace=0.0)
    ax = fig.add_subplot(gs[0,0])
    
    # caratteri
    nk = spinor_distribution.shape[0]

    # bande
    Ekz0 = bands[0,:]
    mi = Ekz0.min()
    ma = Ekz0.max()
    title=''
    
    # spinorial characters
    EL = spinor_distribution[:,0:2,:].sum(axis=1)
    HH = spinor_distribution[:,2:4,:].sum(axis=1)
    LH = spinor_distribution[:,4:6,:].sum(axis=1)
    SO = spinor_distribution[:,6:8,:].sum(axis=1)
    if character_to_show is not None:
        if character_to_show=='EL':
            color = EL
            cmap='Blues'
            ticklabels = ['0', '1']
            ticks=[0,1]
            norm5 = plt.Normalize(0., 1.0)
        elif character_to_show=='HH':
            color= HH
            cmap='Blues'
            ticklabels = ['0', '1']
            ticks=[0,1]
            norm5 = plt.Normalize(0., 1.0)
        elif character_to_show=='LH':
            color= LH
            cmap='Blues'
            ticklabels = ['0', '1']
            ticks=[0,1]
            norm5 = plt.Normalize(0., 1.0)
        elif character_to_show=='SO':
            color= SO
            cmap='Blues'
            ticklabels = ['0', '1']
            ticks=[0,1]
            norm5 = plt.Normalize(0., 1.0)
        elif character_to_show=='LH-HH':
            color= LH - HH
            cmap='viridis'
            ticklabels = ['HH', 'LH']
            ticks=[-0.9,0.9]
            norm5 = plt.Normalize(-1., 1.0)
        elif character_to_show=='H-EL':
            color= (HH+LH+SO)-EL
            cmap='jet'
            ticklabels = ['EL', 'H']
            ticks=[-1,1]
            norm5 = plt.Normalize(-1., 1.0)
        elif character_to_show=='H-EL-reg':
            if norm_sum_region is None:
                raise ValueError("Must provide norm per region vector")
            color= norm_sum_region[:,1,:] - norm_sum_region[:,0,:]
            cmap='jet'
            ticklabels = ['EL', 'H']
            ticks=[-1,1]
            norm5 = plt.Normalize(-1., 1.0)  
        elif character_to_show=='H-EL-reg-threshold':
            if norm_sum_region is None:
                raise ValueError("Must provide norm per region vector")
            if threshold_el is None:
                raise ValueError("Must provide electron threshold")
            if threshold_h is None:
                raise ValueError("Must provide hole threshold")
            color = np.zeros((norm_sum_region.shape[0],norm_sum_region.shape[2]))
            color[norm_sum_region[:,1,:]>threshold_h] = 1.0 #pure holes
            color[norm_sum_region[:,0,:]>threshold_el] = -1.0 #pure electrons
            cmap='jet'
            ticklabels = ['EL', 'H']
            ticks=[-1,1]
            norm5 = plt.Normalize(-1., 1.0) 
        elif character_to_show=="am":
            color = angular_momentum[0]
            label = angular_momentum[1]
            cmap=cmap_in
            title = label #r"$|<F_{z}>|=$"+label
            ticklabels = ["0", "1"]
            ticks=[0.0, 1.0]
            norm5 = plt.Normalize(0., 1.)
        else:
            raise ValueError("invalid character string")
    else:
        color = EL+HH+LH+SO
        cmap='Blues'

    xlim_l = xlim[0]
    xlim_r = xlim[1]
    ylim_l = ylim[0]
    ylim_r = ylim[1]
    ax.set_xlim(xlim_l,xlim_r)
    ax.set_ylim(ylim_l,ylim_r) 

    ax.set_ylabel('$E$ [meV]', size = 20)
    ax.tick_params('y', labelsize=15)
    ax.tick_params('x', labelsize=15)
    for j in range(0,spinor_distribution.shape[2],1):
            
        x = kzvals
        y = bands[:,j]
        
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1) 
        #norm5 = plt.Normalize(0., 1.0)
        lc = LineCollection(segments, cmap=cmap, norm=norm5)
        lc.set_array(color[:,j])
        lc.set_linewidth(4)
        line5 = ax.add_collection(lc)
        
    cbaxes = inset_axes(ax, width="40%", height="6%", loc=1) 
    
    cbar = fig.colorbar(line5,
                 cax=cbaxes,
                 ticks=ticks,
                 orientation='horizontal')
    
    cbar.ax.set_xticklabels(ticklabels,size=15)
    cbar.ax.tick_params(size=0)
    #cbar.ax.set_title(title, fontsize=20, loc="right")
    cbar.ax.set_xlabel(xlabel=title, fontsize=20)

    # plot chemical potentials
    if chemical_potential is not None:
        ax.axhline(y=chemical_potential, color='black', ls='--', lw=4)
        #ax.text(0.08,chemical_potential+0.003,r"$\mu$",size=15)
    
    ax.set_xlabel('$k_{z}[nm^{-1}]$',size=20)
    return fig

def plot_bands_scatter(
        kzvals, 
        bands, 
        angular_momentum = None,  
        chemical_potential=None, 
        figsize=(5, 5), 
        xlim=None, 
        ylim=None, 
        cmap_in='Blues'
    ):
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(1, 1, left=0.00, right=1, top=1.0, bottom=0.0, hspace=0.0)
    ax = fig.add_subplot(gs[0,0])

    color = angular_momentum[0]
    label = angular_momentum[1]
    cmap=cmap_in
    title = label #r"$|<F_{z}>|=$"+label
    ticklabels = ["0", "1"]
    ticks=[0.0, 1.0]
    norm5 = plt.Normalize(0., 1.)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_ylabel('$E$ [meV]', size = 20)
    ax.tick_params('y', labelsize=15)
    ax.tick_params('x', labelsize=15)

    for j in range(0,bands.shape[1],1):
            
        x = kzvals
        y = bands[:,j]

        scplot = ax.scatter(x=x, y=y, s=color[:,j]*30., c=color[:,j], cmap=cmap_in, norm=norm5, marker='o', alpha=0.8)
        ax.plot(x, y,ls='-',lw=1, color='black', alpha=0.8)
        
    cbaxes = inset_axes(ax, width="40%", height="6%", loc=1) 
    
    cbar = fig.colorbar(scplot,
                 cax=cbaxes,
                 ticks=ticks,
                 orientation='horizontal')
    
    cbar.ax.set_xticklabels(ticklabels,size=15)
    cbar.ax.tick_params(size=0)
    #cbar.ax.set_title(title, fontsize=20, loc="right")
    cbar.ax.set_xlabel(xlabel=title, fontsize=25)

    # plot chemical potentials
    if chemical_potential is not None:
        ax.axhline(y=chemical_potential, color='black', ls='--', lw=4)
        #ax.text(0.08,chemical_potential+0.003,r"$\mu$",size=15)
    
    #ax.set_xlabel('$ k_{z} ( \\frac{\pi}{\sqrt{3} a_{lc}} )$',size=20)   
    ax.set_xlabel('$ k_{z} [nm^{-1}]$',size=20)

    return fig



def plot_density_and_potential_1D(*density, potential=None, dslice=100.0):
    
    fig = plt.figure(figsize=(7, 7))
    gs = gridspec.GridSpec(1, 1, left=0.00, right=1, top=1.0, bottom=0.0, hspace=0.0) 
    ax = fig.add_subplot(gs[0,0])

    ptee = np.linspace(-dslice,dslice,600)
    #xcc = ptee*np.cos(np.pi/6)
    #ycc = (ptee*np.sin(np.pi/6))[::-1]
    xcc = np.linspace(-dslice,dslice,600)
    ycc = np.zeros(600)
    
    xee = np.zeros(600)
    yee = np.linspace(-dslice,dslice,600)
    
    coords_cc = np.vstack([xcc,ycc]).T
    coords_ee = np.vstack([xee,yee]).T

    n_lin_cc = np.zeros(600)
    n_lin_ee = np.zeros(600)
    
    # densità di carica
    epot = potential

    for rho in density:   
        n_lin_cc += rho.interp(coords_cc / _constants.length_scale) / 1e16 
        n_lin_ee += rho.interp(coords_ee / _constants.length_scale) / 1e16
    
    if epot is not None: 
        V_lin_cc = epot.interp(coords_cc / _constants.length_scale) #+ ev_cc
        V_lin_ee = epot.interp(coords_ee / _constants.length_scale) #+ ev_ee
    else:
        V_lin_cc = np.zeros(len(coords_cc))
        V_lin_ee = np.zeros(len(coords_ee))
    #ax.set_title('Charge density')
    # Make the y-axis label, ticks and tick labels match the line color.
    #ax.set_ylabel('$n_{el} + n_{h} + n_{dop}$ $[10^{16} \mathrm{cm}^{-3}]$', size = 15, color='b')
    ax.set_ylabel('$n_{h}$ $[10^{16} \mathrm{cm}^{-3}]$', size = 20, color='b')
    ax.set_xlabel('position [nm]', size = 20, color='black')
    ax.tick_params('y', labelsize=15, colors='b')
    ax.tick_params('x', labelsize=15)
    ax2 = ax.twinx()

    ax.plot(xcc*1e-1, n_lin_cc, label='cc', color='b', ls='-', alpha=0.8)
    ax.plot(yee*1e-1, n_lin_ee, label='ee', color='b', ls='--')        
    #ax2.plot(xcc*1e-1, V_lin_cc, color='r', ls='-',alpha=0.8)
    ax2.plot(ptee*1e-1, V_lin_cc, color='r', ls='-',alpha=0.8)
    ax2.plot(yee*1e-1, V_lin_ee, color='r', ls='--')
    #ax2.set_ylabel('- VB [meV]', size = 15, color='r')
    ax2.set_ylabel('$-eV_{el}$ [meV]', size = 20, color='r')
    ax2.tick_params('y', labelsize=15, colors='r')
        
    #ax0[i].set_yticks(np.linspace(0,np.round(np.max(n_lin_cc),2),5))
    ax2.set_ylim(-0.012,0.012)
    ax.yaxis.set_major_locator(MaxNLocator(5)) 
    ax2.yaxis.set_major_locator(MaxNLocator(6))

    # plot vertical lines
    #for key, vline in vlines.items():
    #    ax.axvline(x=vline, color='black', ls='-', lw=2)


def plot_density_and_potential_2D(*density, potential=None, mesh,subdiv=1, **polygons):

    fig = plt.figure(figsize=(12, 5))
    gs = gridspec.GridSpec(1, 2, left=0.00, right=1, top=1.0, bottom=0.0, hspace=0.7) 
    ax1 = fig.add_subplot(gs[0,0])
    ax2 = fig.add_subplot(gs[0,1])

    triangles = mesh.triangles[:,:3]
    x = mesh.vertices[:,0] #*staticdata.br / 10 # nm
    y = mesh.vertices[:,1] #*staticdata.br / 10 # nm
    trigrid = tri.Triangulation(x, y, triangles )
    ref_trigrid = UniformTriRefiner(trigrid)
    trigrid = ref_trigrid.refine_triangulation(subdiv=subdiv)
    vert = np.vstack([trigrid.x, trigrid.y]).T
    triangles = trigrid.triangles

    V = np.zeros(len(vert))
    if potential is not None:
        V = potential.interp(vert) #* 1e3
    
    ch = np.zeros(len(vert))
    for rho in density:
        ch += rho.interp(vert) / 1e16
    
    pot2D = ax1.tricontourf(
        vert[:,0] * _constants.length_scale / 10, 
        vert[:,1] * _constants.length_scale / 10,
        triangles,
        V,
        levels=50,
        cmap='viridis'
    )

    ax1.tick_params('both', labelsize=15)

    
    cbaxes1 = inset_axes(ax1, width="2%", height="90%",loc='upper left',
                    bbox_to_anchor=(1.,1-0.5,.5,.5), bbox_transform=ax1.transAxes) 
    
    cbar1 = fig.colorbar(pot2D,
                 cax=cbaxes1,
                 orientation='vertical')
    cbar1.ax.tick_params(labelsize=15)
    
    ch2D = ax2.tricontourf(
        vert[:,0] * _constants.length_scale / 10, 
        vert[:,1] * _constants.length_scale / 10,
        triangles,
        ch,
        levels=50,
        cmap='viridis'
    )

    ax2.tick_params('both', labelsize=15)
    
    cbaxes2 = inset_axes(ax1, width="2%", height="90%",loc='upper left',
                    bbox_to_anchor=(1.,1-0.5,.5,.5), bbox_transform=ax2.transAxes) 
    
    cbar2 = fig.colorbar(ch2D,
                 cax=cbaxes2,
                 orientation='vertical')

    cbar2.ax.tick_params(labelsize=15)

    # plot region boundaries

    for key, poly in polygons.items():
        newpoly = copy.copy(poly)
        ax1.add_patch(newpoly)
        newpoly = copy.copy(poly)
        ax2.add_patch(newpoly)

    ax1.set_title(r'$-eV_{el}$ [meV]', fontsize=20)
    ax2.set_title(r'$n_{h}$ $[10^{16} \mathrm{cm}^{-3}]$', fontsize=20)

    ax1.set_xlabel('position [nm]', size = 20, color='black')
    ax1.set_ylabel('position [nm]', size = 20, color='black')
    ax2.set_xlabel('position [nm]', size = 20, color='black')

    return fig


def plot_envelope_functions(wf_el, wf_h, spinor_dist, mesh, numax, character_to_show=None, nk=0, subdiv=1, **polygons):

    triangles = mesh.triangles[:,:3]
    x = mesh.vertices[:,0] #*staticdata.br / 10 # nm
    y = mesh.vertices[:,1] #*staticdata.br / 10 # nm
    trigrid = tri.Triangulation(x, y, triangles )
    ref_trigrid = UniformTriRefiner(trigrid)
    trigrid = ref_trigrid.refine_triangulation(subdiv=subdiv)
    vert = np.vstack([trigrid.x, trigrid.y]).T
    triangles = trigrid.triangles

    if character_to_show=='H-EL':
        F_h = wf_h.interp(vert)
        F_el = wf_el.interp(vert)
        #F_el2 = np.add.reduceat(np.abs(F_el)**2, np.arange(0,F_el.shape[-1],2),axis=-1)
        F_el2 = np.abs(F_el)**2
        #F_h2 = np.add.reduceat(np.abs(F_h)**2, np.arange(0,F_h.shape[-1],2),axis=-1)
        F_h2 = np.abs(F_h)**2
        F_h2 = np.divide(F_h2,np.max(F_h2,axis=1)[:,np.newaxis,:])
        F_el2 = np.divide(F_el2,np.max(F_el2,axis=1)[:,np.newaxis,:])
        F_el2_w = np.multiply(F_el2,spinor_dist[:,np.newaxis,:2,:]).sum(axis=2)
        F_h2_w = np.multiply(F_h2,spinor_dist[:,np.newaxis,2:,:]).sum(axis=2)
        F_1 = F_el2_w
        F_2 = F_h2_w
        F = F_1 + F_2
        title1 = r'$\phi_{\mathrm{EL}}$'
        title2 = r'$\phi_{\mathrm{H}}$'
    elif character_to_show=='HH-LH':
        F_h = wf_h.interp(vert)
        #F_h2 = np.add.reduceat(np.abs(F_h)**2, np.arange(0,F_h.shape[-1],2),axis=-1)
        F_h2 = np.abs(F_h)**2
        F_h2 = np.divide(F_h2,np.max(F_h2,axis=1)[:,np.newaxis,:])
        #F_h2_w = np.multiply(F_h2,spinor_dist[:,np.newaxis,2:,::2])
        F_h2_w = np.multiply(F_h2,spinor_dist[:,np.newaxis,2:,:])
        F_1 = F_h2_w[:,:,0:2,:].sum(axis=2) #HH
        F_2 = F_h2_w[:,:,2:4,:].sum(axis=2) #LH
        F = F_1 + F_2
        title1 = r'$\phi_{\mathrm{HH}}$'
        title2 = r'$\phi_{\mathrm{LH}}$'

    
    #nu_vals = (np.arange(1,numax)-1)*2
    nu_vals = np.arange(0,numax)
    nrow = len(nu_vals)
    ncol = 3
    
    fig, ax = plt.subplots(nrow, ncol, sharex='col', sharey='row', figsize=(7, 2*len(nu_vals)))
    l=0
    for nu in nu_vals:
        m = 0
        #lab = int(nu-2*l/2)+1
        lab = nu+1
        ax[l,0].set_ylabel(r'$n$='+str(lab), size=15)
        x = trigrid.x * _constants.length_scale / 10
        y = trigrid.y * _constants.length_scale / 10
        nu = F_h2.shape[-1]-1-nu

        # plot conduction band
        axx = ax[l,0]
        cf = axx.tricontourf(x,
                             y,
                             triangles,
                             F[nk,:,nu],# + F_h2_w[nk,:,nu],
                             50,
                             cmap= 'Blues',
                             alpha=1,
                             vmin=0.0,
                             vmax=1.0)
                
        ax[l,1].tricontourf(x,
                            y,
                            triangles,
                            F_1[nk,:,nu],
                            50,
                            cmap= 'Blues',
                            alpha=1,
                            vmin=0.0,
                            vmax=1.0)
        
        ax[l,2].tricontourf(x,
                            y,
                            triangles,
                            F_2[nk,:,nu],
                            50,
                            cmap= 'Blues',
                            alpha=1,
                            vmin=0.0,
                            vmax=1.0)
        #ax[l,2].triplot(mesh, alpha=0.1)
            
        l+=1
        
    
    cbaxes = ax[-1,1].inset_axes(bounds=[0.0,-0.1,1.0,0.08]) 
    cbar = fig.colorbar(cf,
                 cax=cbaxes,
                 ticks=np.array([0,1.0]),
                 orientation='horizontal')
    #cbar.ax.set_xticklabels(['0', '1'], size=12)
    #cbar.ax.tick_params(size=0)
    
    ax[0,0].set_title(r'$\phi_{\mathrm{TOT}}$', size=25,y=1.05)
    ax[0,1].set_title(title1, size=25,y=1.05)
    ax[0,2].set_title(title2, size=25,y=1.05)
    for axx in ax[:,:].flat:
        axx.yaxis.set_tick_params(which='both', labelleft=False, labelright=False)
        axx.yaxis.offsetText.set_visible(False)
       
        axx.xaxis.set_tick_params(which='both', labelleft=False, labelright=False)
        axx.xaxis.offsetText.set_visible(False) 
        #axx.set_xlim(-r-20,r+20)
        #axx.set_ylim(-420,+420)

        for key, poly in polygons.items():
            newpoly = copy.copy(poly)
            axx.add_patch(newpoly)
        
        axx.set_yticklabels([])
        axx.set_xticklabels([])
        axx.set_xticks([])
        axx.set_yticks([])
        axx.yaxis.label.set_size(25)
        for pos in ['right', 'top', 'bottom', 'left']:
            axx.spines[pos].set_visible(False)
    fig.subplots_adjust(wspace=0, hspace=0)
    return fig

def plot_envelope_functions_3(wf_el, wf_h, symg_h, spinor_dist, mesh, numax, character_to_show=None, nk=0, subdiv=1, **polygons):

    triangles = mesh.triangles[:,:3]
    x = mesh.vertices[:,0] #*staticdata.br / 10 # nm
    y = mesh.vertices[:,1] #*staticdata.br / 10 # nm
    trigrid = tri.Triangulation(x, y, triangles )
    ref_trigrid = UniformTriRefiner(trigrid)
    trigrid = ref_trigrid.refine_triangulation(subdiv=subdiv)
    vert = np.vstack([trigrid.x, trigrid.y]).T
    triangles = trigrid.triangles

    if character_to_show=='H-EL':
        F_h = wf_h.interp(vert)
        F_el = wf_el.interp(vert)
        #F_el2 = np.add.reduceat(np.abs(F_el)**2, np.arange(0,F_el.shape[-1],2),axis=-1)
        F_el2 = np.abs(F_el)**2
        #F_h2 = np.add.reduceat(np.abs(F_h)**2, np.arange(0,F_h.shape[-1],2),axis=-1)
        F_h2 = np.abs(F_h)**2
        F_h2 = np.divide(F_h2,np.max(F_h2,axis=1)[:,np.newaxis,:])
        F_el2 = np.divide(F_el2,np.max(F_el2,axis=1)[:,np.newaxis,:])
        F_el2_w = np.multiply(F_el2,spinor_dist[:,np.newaxis,:2,:]).sum(axis=2)
        F_h2_w = np.multiply(F_h2,spinor_dist[:,np.newaxis,2:,:]).sum(axis=2)
        F_1 = F_el2_w
        F_2 = F_h2_w
        F = F_1 + F_2
        title1 = r'$\phi_{\mathrm{EL}}$'
        title2 = r'$\phi_{\mathrm{H}}$'
    elif character_to_show=='HH-LH':
        F_h = wf_h.project_irrep(symg_h, vert)
        #F_h = wf_h.interp(vert)
        F_h2 = np.add.reduceat(np.abs(F_h)**2, np.arange(0,F_h.shape[-1],2),axis=-1)
        F_h2 = np.divide(F_h2,np.max(F_h2,axis=1)[:,np.newaxis,:])
        F_h2_w = np.multiply(F_h2,spinor_dist[:,np.newaxis,2:,::2])
        F_1 = F_h2_w[:,:,0:2,:].sum(axis=2) #HH
        F_2 = F_h2_w[:,:,2:4,:].sum(axis=2) #LH
        F = F_1 + F_2
        title1 = r'$\phi_{\mathrm{HH}}$'
        title2 = r'$\phi_{\mathrm{LH}}$'

    
    #nu_vals = (np.arange(1,numax)-1)*2
    nu_vals = np.arange(0,numax)
    nrow = len(nu_vals)
    ncol = 3
    
    fig, ax = plt.subplots(nrow, ncol, sharex='col', sharey='row', figsize=(7, 2*len(nu_vals)))
    l=0
    for nu in nu_vals:
        m = 0
        #lab = int(nu-2*l/2)+1
        lab = nu+1
        ax[l,0].set_ylabel(r'$n$='+str(lab), size=15)
        x = trigrid.x * _constants.length_scale / 10
        y = trigrid.y * _constants.length_scale / 10
        nu = F_h2.shape[-1]-1-nu

        # plot conduction band
        axx = ax[l,0]
        cf = axx.tricontourf(x,
                             y,
                             triangles,
                             F[nk,:,nu],# + F_h2_w[nk,:,nu],
                             50,
                             cmap= 'Blues',
                             alpha=1,
                             vmin=0.0,
                             vmax=1.0)
                
        ax[l,1].tricontourf(x,
                            y,
                            triangles,
                            F_1[nk,:,nu],
                            50,
                            cmap= 'Blues',
                            alpha=1,
                            vmin=0.0,
                            vmax=1.0)
        
        ax[l,2].tricontourf(x,
                            y,
                            triangles,
                            F_2[nk,:,nu],
                            50,
                            cmap= 'Blues',
                            alpha=1,
                            vmin=0.0,
                            vmax=1.0)
        #ax[l,2].triplot(mesh, alpha=0.1)
            
        l+=1
        
    
    cbaxes = ax[-1,1].inset_axes(bounds=[0.0,-0.1,1.0,0.08]) 
    cbar = fig.colorbar(cf,
                 cax=cbaxes,
                 ticks=np.array([0,1.0]),
                 orientation='horizontal')
    #cbar.ax.set_xticklabels(['0', '1'], size=12)
    #cbar.ax.tick_params(size=0)
    
    ax[0,0].set_title(r'$\phi_{\mathrm{TOT}}$', size=25,y=1.05)
    ax[0,1].set_title(title1, size=25,y=1.05)
    ax[0,2].set_title(title2, size=25,y=1.05)
    for axx in ax[:,:].flat:
        axx.yaxis.set_tick_params(which='both', labelleft=False, labelright=False)
        axx.yaxis.offsetText.set_visible(False)
       
        axx.xaxis.set_tick_params(which='both', labelleft=False, labelright=False)
        axx.xaxis.offsetText.set_visible(False) 
        #axx.set_xlim(-r-20,r+20)
        #axx.set_ylim(-420,+420)

        for key, poly in polygons.items():
            newpoly = copy.copy(poly)
            axx.add_patch(newpoly)
        
        axx.set_yticklabels([])
        axx.set_xticklabels([])
        axx.set_xticks([])
        axx.set_yticks([])
        axx.yaxis.label.set_size(25)
        for pos in ['right', 'top', 'bottom', 'left']:
            axx.spines[pos].set_visible(False)
    fig.subplots_adjust(wspace=0, hspace=0)
    return fig

def plot_envelope_functions_2(wf_el, wf_h, mesh, numax, nk=0, subdiv=1, **polygons):

    triangles = mesh.triangles[:,:3]
    x = mesh.vertices[:,0] #*staticdata.br / 10 # nm
    y = mesh.vertices[:,1] #*staticdata.br / 10 # nm
    trigrid = tri.Triangulation(x, y, triangles )
    ref_trigrid = UniformTriRefiner(trigrid)
    trigrid = ref_trigrid.refine_triangulation(subdiv=subdiv)
    vert = np.vstack([trigrid.x, trigrid.y]).T
    triangles = trigrid.triangles

    F_h = wf_h.interp(vert)
    F_el = wf_el.interp(vert)
    F_el2 = np.abs(F_el)**2
    F_h2 = np.abs(F_h)**2
    F_h2 = np.divide(F_h2,np.max(F_h2,axis=1)[:,np.newaxis,:])
    F_el2 = np.divide(F_el2,np.max(F_el2,axis=1)[:,np.newaxis,:])
    #F_el2_w = np.multiply(F_el2,spinor_dist[:,np.newaxis,:2,::2]).sum(axis=2)
    #F_h2_w = np.multiply(F_h2,spinor_dist[:,np.newaxis,2:,::2]).sum(axis=2)

    
    #nu_vals = (np.arange(1,numax)-1)*2
    nu_vals = np.arange(0,numax)
    nrow = len(nu_vals)
    ncol = 8
    
    fig, ax = plt.subplots(nrow, ncol, sharex='col', sharey='row', figsize=(14, 2*len(nu_vals)))
    i=0
    l=0
    for nu in nu_vals:
        m = 0
        #lab = int(nu-2*l/2)+1
        lab = nu+1
        ax[l,0].set_ylabel(r'$n$='+str(lab), size=15)
        x = trigrid.x * _constants.length_scale / 10
        y = trigrid.y * _constants.length_scale / 10
        nu = F_h2.shape[-1]-1-nu

        # plot conduction band
        for i in range(2):
            axx = ax[l,i]
            cf = axx.tricontourf(
                x,
                y,
                triangles,
                F_el2[nk,:,i,nu],
                50,
                cmap= 'Blues',
                alpha=1,
                vmin=0.0,
                vmax=1.0
            )
        for i in range(6):
            axx = ax[l,i+2]
            cf = axx.tricontourf(
                x,
                y,
                triangles,
                F_h2[nk,:,i,nu],
                50,
                cmap= 'Blues',
                alpha=1,
                vmin=0.0,
                vmax=1.0
            )                        
        l+=1
        
    
    cbaxes = ax[-1,1].inset_axes(bounds=[0.0,-0.1,1.0,0.08]) 
    cbar = fig.colorbar(cf,
                 cax=cbaxes,
                 ticks=np.array([0,0.9]),
                 orientation='horizontal')
    #cbar.ax.set_xticklabels(['0', '1'], size=12)
    #cbar.ax.tick_params(size=0)
    
    ax[0,0].set_title(r'$\phi_{\mathrm{EL} \uparrow }$', size=25,y=1.05)
    ax[0,1].set_title(r'$\phi_{\mathrm{EL} \downarrow }$', size=25,y=1.05)
    ax[0,2].set_title(r'$\phi_{\mathrm{HH} \uparrow }$', size=25,y=1.05)
    ax[0,3].set_title(r'$\phi_{\mathrm{HH} \downarrow }$', size=25,y=1.05)
    ax[0,4].set_title(r'$\phi_{\mathrm{LH}  \uparrow }$', size=25,y=1.05)
    ax[0,5].set_title(r'$\phi_{\mathrm{LH} \downarrow }$', size=25,y=1.05)
    ax[0,6].set_title(r'$\phi_{\mathrm{SO} \uparrow }$', size=25,y=1.05)
    ax[0,7].set_title(r'$\phi_{\mathrm{SO} \downarrow }$', size=25,y=1.05)
    for axx in ax[:,:].flat:
        axx.yaxis.set_tick_params(which='both', labelleft=False, labelright=False)
        axx.yaxis.offsetText.set_visible(False)
       
        axx.xaxis.set_tick_params(which='both', labelleft=False, labelright=False)
        axx.xaxis.offsetText.set_visible(False) 
        #axx.set_xlim(-r-20,r+20)
        #axx.set_ylim(-420,+420)

        for key, poly in polygons.items():
            newpoly = copy.copy(poly)
            axx.add_patch(newpoly)
        
        axx.set_yticklabels([])
        axx.set_xticklabels([])
        axx.set_xticks([])
        axx.set_yticks([])
        axx.yaxis.label.set_size(25)
        for pos in ['right', 'top', 'bottom', 'left']:
            axx.spines[pos].set_visible(False)
    fig.subplots_adjust(wspace=0, hspace=0)
    return fig

def plot_bands_cioni(kzvals, bands, spinor_distribution, character_to_show=None, figsize=(5, 5), xlim=(0.0, 0.1), ylim=(-0.1,0.0)):
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(1, 1, left=0.00, right=1, top=1.0, bottom=0.0, hspace=0.0)
    ax = fig.add_subplot(gs[0,0])
    
    # caratteri
    nk = spinor_distribution.shape[0]

    # bande
    Ekz0 = bands[0,:]
    
    X = spinor_distribution[:,0:2,:].sum(axis=1)
    Y = spinor_distribution[:,2:4,:].sum(axis=1)
    Z = spinor_distribution[:,4:6,:].sum(axis=1)
    if character_to_show is not None:
        if character_to_show=='X':
            color = X
        elif character_to_show=='Y':
            color= Y
        elif character_to_show=='Z':
            color= Z
        else:
            raise ValueError("invalid character string")
    else:
        color = X + Y + Z
    # struttura a bande con colori per carattere
    ax.set_xlim(xlim[0], xlim[1])
    ax.set_ylim(ylim[0], ylim[1])
    ax.set_ylabel('$E$ [meV]', size = 20)
    for j in range(0,spinor_distribution.shape[2],1):
            
        x = kzvals
        y = bands[:,j]
        
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1) 
        norm5 = plt.Normalize(0., 1.0)
        lc = LineCollection(segments, cmap= 'Blues', norm=norm5)
        lc.set_array(color[:,j])
        lc.set_linewidth(4)
        line5 = ax.add_collection(lc)
        
    #cbaxes = inset_axes(ax, width="40%", height="6%", loc=1) 
    
    #cbar = fig.colorbar(line5,
    #             cax=cbaxes,
    #             ticks=[0,1],
    #             orientation='horizontal')
    #
    #cbar.ax.set_xticklabels(['0', '1'],size=15)
    #cbar.ax.tick_params(size=0)
    
    ax.set_xlabel('$ k_{z} [nm^{-1}]$',size=20)    
    return fig


def plot_envelope_functions_cioni(wf_h, spinor_distribution, mesh, numax, nk=0, subdiv=1, **polygons):

    triangles = mesh.triangles[:,:3]
    x = mesh.vertices[:,0] #*staticdata.br / 10 # nm
    y = mesh.vertices[:,1] #*staticdata.br / 10 # nm
    trigrid = tri.Triangulation(x, y, triangles )
    ref_trigrid = UniformTriRefiner(trigrid)
    trigrid = ref_trigrid.refine_triangulation(subdiv=subdiv)
    vert = np.vstack([trigrid.x, trigrid.y]).T
    triangles = trigrid.triangles

    wf = wf_h.interp(vert)

    wfsq = np.abs(wf)**2
    wfsq_tot = wfsq.sum(axis=2)
    wfsq_tot = np.divide(wfsq_tot,np.max(wfsq_tot,axis=1)[:,np.newaxis,:])
    wfsq = np.divide(wfsq,np.max(wfsq,axis=1)[:,np.newaxis,:])
    wfsq_w = np.multiply(wfsq, spinor_distribution[:,np.newaxis,:,:])
    
    #nu_vals = (np.arange(1,numax)-1)*2
    nu_vals = np.arange(0,numax)
    nrow = len(nu_vals)
    ncol = 7
    
    fig, ax = plt.subplots(nrow, ncol, sharex='col', sharey='row', figsize=(13, 2*len(nu_vals)))
    i=0
    l=0
    for nu in nu_vals:
        m = 0
        #lab = int(nu-2*l/2)+1
        lab = nu+1
        ax[l,0].set_ylabel(r'$n$='+str(lab), size=15)
        x = trigrid.x * 1e9 # nm
        y = trigrid.y * 1e9 # nm
        nu = wfsq.shape[-1]-1-nu

        # plot conduction band
        axx = ax[l,0]
        cf = axx.tricontourf(
            x,
            y,
            triangles,
            wfsq_tot[nk,:,nu],
            50,
            cmap= 'Blues',
            alpha=1,
            vmin=0.0,
            vmax=1.0
        )
        for i in range(6):
            axx = ax[l,i+1]
            cf1 = axx.tricontourf(
                x,
                y,
                triangles,
                wfsq_w[nk,:,i,nu],
                50,
                cmap= 'Blues',
                alpha=1,
                vmin=0.0,
                vmax=1.0
            )                        
        l+=1
        
    
    cbaxes = ax[-1,3].inset_axes(bounds=[0.0,-0.1,1.0,0.08]) 
    cbar = fig.colorbar(cf,
                 cax=cbaxes,
                 ticks=[0,1],
                 orientation='horizontal')
    cbar.ax.set_xticklabels(['0', '1'], size=12)
    #cbar.ax.tick_params(size=0)
    
    ax[0,0].set_title(r'$\phi_{\mathrm{tot}}$', size=25,y=1.05)
    ax[0,1].set_title(r'$\phi_{\mathrm{X} \uparrow }$', size=25,y=1.05)
    ax[0,2].set_title(r'$\phi_{\mathrm{Y} \uparrow }$', size=25,y=1.05)
    ax[0,3].set_title(r'$\phi_{\mathrm{Z} \uparrow }$', size=25,y=1.05)
    ax[0,4].set_title(r'$\phi_{\mathrm{X} \downarrow }$', size=25,y=1.05)
    ax[0,5].set_title(r'$\phi_{\mathrm{Y} \downarrow }$', size=25,y=1.05)
    ax[0,6].set_title(r'$\phi_{\mathrm{Z} \downarrow }$', size=25,y=1.05)
    #ax[0,0].set_title(r'$\phi_{\mathrm{tot}}$', size=25,y=1.05)
    #ax[0,1].set_title(r'$\phi_{\mathrm{HH} \uparrow }$', size=25,y=1.05)
    #ax[0,2].set_title(r'$\phi_{\mathrm{HH} \uparrow }$', size=25,y=1.05)
    #ax[0,3].set_title(r'$\phi_{\mathrm{LH} \uparrow }$', size=25,y=1.05)
    #ax[0,4].set_title(r'$\phi_{\mathrm{LH} \downarrow }$', size=25,y=1.05)
    #ax[0,5].set_title(r'$\phi_{\mathrm{SO} \downarrow }$', size=25,y=1.05)
    #ax[0,6].set_title(r'$\phi_{\mathrm{SO} \downarrow }$', size=25,y=1.05)
    for axx in ax[:,:].flat:
        axx.yaxis.set_tick_params(which='both', labelleft=False, labelright=False)
        axx.yaxis.offsetText.set_visible(False)
       
        axx.xaxis.set_tick_params(which='both', labelleft=False, labelright=False)
        axx.xaxis.offsetText.set_visible(False) 
        #axx.set_xlim(-r-20,r+20)
        #axx.set_ylim(-420,+420)

        for key, poly in polygons.items():
            newpoly = copy.copy(poly)
            axx.add_patch(newpoly)
        
        axx.set_yticklabels([])
        axx.set_xticklabels([])
        axx.set_xticks([])
        axx.set_yticks([])
        axx.yaxis.label.set_size(25)
        for pos in ['right', 'top', 'bottom', 'left']:
            axx.spines[pos].set_visible(False)
    fig.subplots_adjust(wspace=0, hspace=0)
    return fig


def plot_occupation(bands, fermi_electrons, fermi_holes, chemical_potential, nk=0):
    fig = plt.figure(figsize=(7, 7))
    gs = gridspec.GridSpec(1, 1, left=0.00, right=1, top=1.0, bottom=0.0, hspace=0.0) 
    ax = fig.add_subplot(gs[0,0])

    ax.scatter(bands[nk,:], fermi_electrons[0,:], color='blue', marker='o', s=15, label='Electrons')

    ax.scatter(bands[nk,:], fermi_holes[0,:], color='green', marker='o', s=15, label='Holes') 

    ax.axvline(x=chemical_potential, color='black', ls='--', lw=2)
    ax.set_xlabel('$E$ [eV]', size = 20)
    ax.set_ylabel('Occupation', size = 20)
    ax.legend(loc=0, fontsize=20)

    return fig

def plot_presid(presid_rel, presid):
    fig = plt.figure(figsize=(7, 7))
    gs = gridspec.GridSpec(1, 1, left=0.00, right=1, top=1.0, bottom=0.0, hspace=0.0) 
    ax = fig.add_subplot(gs[0,0])
    ax.set_ylabel('$p_{resid}/p$', size = 18, color='black')
    ax.set_xlabel('Iterations', size = 18, color='black')
    ax.plot(presid_rel)
    
    ax2 = ax.twinx()
    ax.plot(np.arange(presid_rel.shape[0]), presid_rel, color='black', ls='-', alpha=0.8, lw=5)
    ax2.plot(np.arange(presid_rel.shape[0]), presid/1e7, color='blue', ls='-',alpha=0.8, lw=5)
    ax2.set_ylabel('$p_{resid}$ $[10^{7} \mathrm{cm}^{-1}]$', size = 18, color='blue')
    ax2.tick_params('y', colors='blue')
    ax.tick_params('y', colors='black', labelsize=15) 
    ax.tick_params('x', colors='black', labelsize=15) 
    ax2.tick_params('y', colors='blue', labelsize=15)  
    ax.set_xticks(np.arange(presid_rel.shape[0]))
    return fig

def plot_total_charge_vs_chempot(total_charge, muvals):
    fig = plt.figure(figsize=(7, 7))
    gs = gridspec.GridSpec(1, 1, left=0.00, right=1, top=1.0, bottom=0.0, hspace=0.0) 
    ax = fig.add_subplot(gs[0,0])
    ax.set_ylabel('$n + p$ $[10^{7} \mathrm{cm}^{-1}]$', size = 18, color='black')
    ax.set_xlabel('Chemical potential [meV]', size = 18, color='black')
    
    ax.plot(muvals, total_charge/1e7, color='black', ls='-', alpha=0.8, lw=5)
    
    ax.tick_params('y', colors='black', labelsize=15) 
    ax.tick_params('x', colors='black', labelsize=15) 
    ax.set_xticks(muvals)
    return fig
