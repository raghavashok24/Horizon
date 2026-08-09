"""
Calibrated N-bank RTGS environment, latent-state estimator, and rollout generator.

All constants are sourced (see comments). Two correctness features:
  * censoring correction — settlement delay is value-weighted with unsettled value
    counted at the end-of-day horizon (averaging settled payments only is biased).
  * realized-delay estimator — waits record the realized delay of SETTLED payments;
    theta_hat = 1 - 1/(1 + mean(delay)) is a self-normalizing geometric MLE.

Public API: Env, simulate(...), theta_hat(...).
"""
import numpy as np
# ============ CALIBRATED CONSTANTS — every value sourced ============
# Fedwire intraday value shares: Armantier, Arnold & McAndrews (2008),
# FRBNY EPR 14(2) Table 1, 2006 decile times inverted by monotone interpolation.
FED=[0.0238,0.0296,0.0516,0.0563,0.0546,0.0744,0.1265,0.2697,0.2827,0.0309]
# FDIC BankFind Call Reports (latest quarter): log-asset fit and cash ratio
LOG_MU,LOG_SD,CASH=13.02,1.56,0.0694
# Fed PSR policy: net debit caps for self-assessed institutions are set as a
# multiple of capital; 0.35 of opening balance is within the documented range.
CAP=0.35
# Treasury collateral haircut ~2% (Fed margining schedule, marketable Treasuries)
HAIRCUT=1.02
T0,N_BANKS,DV=16,50,1e6

class Env:
    def __init__(s,N=N_BANKS,seed=0,reserve_mult=1.0,sph=4):
        r=np.random.default_rng(seed); s.N=N
        s.size=np.exp(r.normal(LOG_MU,LOG_SD,N)); s.size/=s.size.sum()
        s.T=len(FED)*sph; s.prof=np.repeat(np.array(FED)/sph,sph)
        s.bal0=s.size*1e6*CASH*reserve_mult*N
        M=np.outer(s.size,s.size); np.fill_diagonal(M,0); s.CP=M/M.sum(1,keepdims=True)

def simulate(seed,mult,apath,t0=T0,sh_t=None,sh_b=None,surge=1.0,collect_obs=True):
    e=Env(seed=seed,reserve_mult=mult); N=e.N; ego=int(np.argmax(e.size))
    bal=e.bal0.copy(); q=[[] for _ in range(N)]
    hist,waits,out_h=[],[],[]; own=full=None
    wsum=tot=od_int=coll_int=0.0
    for t in range(e.T):
        arr=e.size*DV*N*e.prof[t]
        if sh_t is not None and t>=sh_t: arr=arr*surge
        for i in range(N):
            if arr[i]>0: q[i].append([arr[i],t]); tot+=arr[i]
        qv=np.array([sum(p[0] for p in x) for x in q])
        if t==t0 and collect_obs:
            full=np.r_[qv/(DV*N),np.maximum(bal,0)/(DV*N)]
            w=np.array(waits) if waits else np.array([0.]); hh=np.array(hist) if hist else np.array([0.])
            own=np.r_[qv[ego]/(DV*N),max(bal[ego],0)/(DV*N),len(q[ego]),
                np.mean([t-p[1] for p in q[ego]]) if q[ego] else 0.,
                w.mean(),w.std(),np.percentile(w,90),len(w),
                np.mean(out_h) if out_h else 0.,hh[-4:].mean(),hh.std(),
                (hh[-4:].mean()-hh[:4].mean()) if len(hh)>=8 else 0.]
        aa=np.full(N,0.5 if t<t0 else apath[min(t-t0,len(apath)-1)]); infl=np.zeros(N)
        for i in range(N):
            if sh_b is not None and sh_t is not None and i==sh_b and t>=sh_t: continue
            budget=min(aa[i]*qv[i],max(bal[i],0)+CAP*e.bal0[i]); rel=0.;keep=[]
            for p in sorted(q[i],key=lambda p:p[1]):
                if rel+p[0]<=budget:
                    rel+=p[0]; wsum+=(t-p[1])*p[0]
                    # FIX: record REALIZED settlement delay of the payment that
                    # actually settled — not the age of the unsettled remainder.
                    if i==ego and t<t0: waits.append(t-p[1])
                else: keep.append(p)
            q[i]=keep; bal[i]-=rel
            if i==ego and t<t0: out_h.append(rel/(e.size[ego]*DV*N+1e-9))
            if rel>0: infl+=rel*e.CP[i]
        bal+=infl
        od_int+=np.maximum(-bal,0).sum()
        coll_int+=HAIRCUT*np.maximum(e.bal0-bal,0).sum()   # collateral tied up by early draw
        if t<t0: hist.append(infl[ego]/(e.size[ego]*DV*N+1e-9))
    qv=np.array([sum(p[0] for p in x) for x in q]); unset=qv.sum()
    return dict(H=np.array(hist),OWN=own,FULL=full,WAITS=np.array(waits),
        vw_delay=(wsum+unset*e.T)/max(tot,1e-9),unsettled=unset/max(tot,1e-9),
        od=od_int/max(tot,1e-9),coll=coll_int/max(tot,1e-9)/e.T)

def theta_hat(waits,rmax=None):
    """FIX: delays are integer step counts -> geometric, not exponential.
    P(settle each step)=p, E[d]=(1-p)/p -> p_hat=1/(1+mean(d)); theta_hat=1-p_hat.
    Self-normalizing: no arbitrary r0."""
    if len(waits)<3: return np.nan
    return float(1-1/(1+np.mean(waits)))
