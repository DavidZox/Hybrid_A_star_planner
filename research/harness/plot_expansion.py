"""研究用：畫出某變體在某頁的節點展開順序。用法：python plot_expansion.py 75 '<變體 JSON>' 3 out.png"""
import json, math, sys
import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import evalsim, base_sim as sim, variant_planner as sim2
th=float(sys.argv[1]); kw=json.loads(sys.argv[2]); s=sys.argv[3]; out=sys.argv[4]
sc=dict(evalsim.SCEN[s]); ox=sc.pop('ox'); k2=dict(kw); k2.update(sc); k2['max_iter']=60000
cm=sim.Costmap(); cm.add_obstacle(ox,0.1,0.2,0.35)
pl=sim2.Planner2(cm,**k2); goal=(1.5,-0.5,0.0)
pl.plan((1.0,1.0,math.radians(th)),goal)
E=np.array([[n.x,n.y,n.t,n.d] for n in pl.expanded])
fig,axs=plt.subplots(1,3,figsize=(21,7))
T=evalsim.TREES[s]
for ax,(lo,hi) in zip(axs,[(0,500),(0,3000),(0,len(E))]):
    e=E[lo:hi]
    sca=ax.scatter(e[:,0],e[:,1],c=np.arange(len(e)),s=2,cmap='viridis')
    ax.plot(T['g'][:,0],T['g'][:,1],'.',color='r',ms=1)
    if pl.found_node:
        P=np.array(pl.chain(pl.found_node)); ax.plot(P[:,0],P[:,1],'k-',lw=1)
    ax.add_patch(plt.Circle((ox,0.1),0.42,fill=False,ls='--',color='r'))
    ax.set_xlim(-0.6,3.1); ax.set_ylim(-2.1,2.6); ax.set_aspect('equal'); ax.set_title('%d..%d of %d'%(lo,hi,len(E)))
    plt.colorbar(sca,ax=ax,fraction=0.03)
plt.tight_layout(); plt.savefig(out,dpi=55)
