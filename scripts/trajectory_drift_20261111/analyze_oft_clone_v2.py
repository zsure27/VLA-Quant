"""Extend v1's unchanged raw audit/math with error-vector mean and RMS scatter."""
import json
import numpy as np
import analyze_oft_clone_v1 as core

base_plots=core.plots


def plots(out,pairs,baselines,suites,conditions):
    base_plots(out,pairs,baselines,suites,conditions)
    import matplotlib.pyplot as plt
    colors=dict(zip(conditions,['#2878b5','#268c70','#7956a3','#e07b39','#b44770']))
    # Replace only the new v2 output's 3D view with isotropic metric axes.
    fig=plt.figure(figsize=(18,5))
    for i,suite in enumerate(suites):
        ax=fig.add_subplot(1,4,i+1,projection='3d');cloud=[]
        if (suite,0) not in baselines:
            ax.set_title(suite+' / missing BF16');continue
        xyz=baselines[suite,0]['position']*1000;cloud.append(xyz)
        ax.plot(*xyz.T,color='#444',label='BF16');ax.scatter(*xyz[0],color='#444',s=12)
        for condition in conditions:
            if (suite,0,condition) not in pairs:continue
            cand,_,t,_,_=pairs[suite,0,condition];xyz=cand['position'][:len(t)]*1000;cloud.append(xyz)
            ax.plot(*xyz.T,color=colors[condition],label=condition)
        points=np.concatenate(cloud);center=(points.min(0)+points.max(0))/2
        half=max(float(np.ptp(points,axis=0).max()/2),1.)
        ax.set_xlim(center[0]-half,center[0]+half);ax.set_ylim(center[1]-half,center[1]+half);ax.set_zlim(center[2]-half,center[2]+half)
        ax.set_box_aspect([1,1,1]);ax.set_title(suite+' / task0 / initial15')
        ax.set_xlabel('Base X (mm)');ax.set_ylabel('Base Y (mm)');ax.set_zlabel('Base Z (mm)')
        if i==0:ax.legend(fontsize=6)
    fig.suptitle('Prespecified task0: observed paired prefixes, equal XYZ scale within each suite')
    fig.savefig(out/'03-prespecified-task0-3d-paths.png',dpi=145,bbox_inches='tight')
    fig.savefig(out/'03-prespecified-task0-3d-paths.pdf',bbox_inches='tight');plt.close(fig)
    fig,axes=plt.subplots(2,4,figsize=(18,8),constrained_layout=True);coverage=[]
    for column,suite in enumerate(suites):
        for condition in conditions:
            rows=[value for key,value in pairs.items() if key[0]==suite and key[2]==condition]
            if not rows:continue
            for axis,kind,factor in [(0,'position',1000),(1,'rotation',180/np.pi)]:
                vectors=[(row[0]['position'][:len(row[2])]-row[1]['position'][:len(row[2])]) if kind=='position' else row[4] for row in rows]
                width=max(len(v) for v in vectors);grid=np.full((len(vectors),width,3),np.nan)
                for i,v in enumerate(vectors):grid[i,:len(v)]=v*factor
                count=np.isfinite(grid[:,:,0]).sum(0);mean=np.nanmean(grid,axis=0)
                bias=np.linalg.norm(mean,axis=1)
                scatter=np.sqrt(np.nansum(np.sum((grid-mean)**2,axis=2),axis=0)/count)
                bias[count<5]=np.nan;scatter[count<5]=np.nan
                x=np.arange(1,width+1);ax=axes[axis,column]
                ax.plot(x,bias,color=colors[condition],label=condition+' mean-vector norm')
                ax.plot(x,scatter,color=colors[condition],ls='--',label=condition+' RMS scatter')
                coverage.append(dict(suite=suite,condition=condition,metric=kind,minimum_tasks=5,contributors=count.tolist()))
        for axis in [0,1]:
            axes[axis,column].set_title(suite+' / initial15');axes[axis,column].set_xlabel('Physical control step; n >= 5')
            axes[axis,column].set_ylabel('Position error-vector bias / scatter (mm)' if axis==0 else 'World SO(3) log-vector bias / scatter (deg)')
            axes[axis,column].grid(alpha=.15)
    axes[0,0].legend(fontsize=5)
    fig.suptitle('Solid: norm of across-task mean error vector; dashed: RMS scatter. Descriptive closed-loop errors.')
    name='05-error-vector-bias-versus-scatter'
    fig.savefig(out/(name+'.png'),dpi=145,bbox_inches='tight');fig.savefig(out/(name+'.pdf'),bbox_inches='tight');plt.close(fig)
    (out/'bias-scatter-coverage.json').write_text(json.dumps(dict(scope='Across-task paired prefixes; not causal or compensation evidence',minimum_contributors=5,coverage=coverage),indent=2)+'\n')
    page=out/'index.html';page.write_text(page.read_text()+'<h2>Error-vector bias and scatter (n >= 5)</h2><img style="width:100%" src="'+name+'.png">')


if __name__=='__main__':
    core.plots=plots
    core.main()
