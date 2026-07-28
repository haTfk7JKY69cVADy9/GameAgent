from pathlib import Path
import sys, shutil, json, subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parent
sys.path.insert(0, str(root))

from model import AgentUtilityParams, DetectorParams, Evidence, GameParams, Signal
from pure_equilibria import OffPathMode, find_pure_equilibria
from mixed_equilibria import find_mixed_equilibria
from beliefs import posterior_compromised
from verifier import best_responses

results_root = root / 'results' / 'boundary_correction_validation'
if results_root.exists():
    shutil.rmtree(results_root)
results_root.mkdir(parents=True)

base = GameParams(); au = base.agent_utility
rho_values = np.linspace(0.0,1.0,21); c_values=np.linspace(0.0,1.0,21)
rows=[]

def classify_pure(eqs):
    desired=any(eq.intact_signal==Signal.REINFORCED and eq.compromised_signal==Signal.BASIC for eq in eqs)
    classes={eq.classification for eq in eqs}
    if desired: return 'separating'
    if 'pooling_basic' in classes and 'pooling_reinforced' in classes: return 'multiple_pooling'
    if 'pooling_basic' in classes: return 'pooling_basic'
    if 'pooling_reinforced' in classes: return 'pooling_reinforced'
    if 'separating' in classes: return 'other_separating'
    return 'none'

for rho in rho_values:
  for cc in c_values:
    p=GameParams(prior_compromised=base.prior_compromised,detector=base.detector,verifier_utility=base.verifier_utility,
      agent_utility=AgentUtilityParams(intact_benefit=au.intact_benefit,compromised_gain=au.compromised_gain,
      intact_reinforced_cost=au.intact_reinforced_cost,compromised_reinforced_cost=float(cc),
      intact_challenge_cost=au.intact_challenge_cost,compromised_challenge_cost=au.compromised_challenge_cost,
      intact_deny_loss=au.intact_deny_loss,compromised_detection_penalty=au.compromised_detection_penalty,
      residual_gain_reinforced=float(rho),residual_gain_challenge_basic=au.residual_gain_challenge_basic,
      residual_gain_challenge_reinforced=au.residual_gain_challenge_reinforced))
    eqs=find_pure_equilibria(p,off_path_mode=OffPathMode.CONSERVATIVE)
    regime=classify_pure(eqs); analytic=cc>=rho*au.compromised_gain+au.compromised_detection_penalty-1e-12
    rows.append({'rho_R':float(rho),'c_C':float(cc),'regime':regime,'number_of_pure_equilibria':len(eqs),
      'computed_separating':regime=='separating','analytic_separation_condition':bool(analytic)})

df=pd.DataFrame(rows); df.to_csv(results_root/'regime_map_corrected.csv',index=False)
counts=df['regime'].value_counts().to_dict(); agreement=float((df.computed_separating==df.analytic_separation_condition).mean())
old_path=Path('/mnt/data/regime_map_results/regime_map_results.csv'); comparison={'old_file_available':old_path.exists()}
if old_path.exists():
  old=pd.read_csv(old_path); col='pure_class' if 'pure_class' in old.columns else 'final_class'
  m=df.merge(old[['rho_R','c_C',col]],on=['rho_R','c_C'],how='left'); comparison.update(changed_cells=int((m.regime!=m[col]).sum()),total_cells=len(m))

codes={n:i for i,n in enumerate(['separating','pooling_basic','pooling_reinforced','multiple_pooling','other_separating','none'])}
pivot=df.pivot(index='c_C',columns='rho_R',values='regime').sort_index(); Z=pivot.replace(codes).to_numpy(dtype=float)
fig,ax=plt.subplots(figsize=(9,6.5)); ax.imshow(Z,origin='lower',aspect='auto',extent=[0,1,0,1],interpolation='nearest')
fr=np.linspace(0,.8,300); ax.plot(fr,fr+.2,linewidth=2,label=r'$c_C=\rho_R G_C+r_C$'); ax.set_xlabel(r'$\rho_R$'); ax.set_ylabel(r'$c_C$'); ax.set_title('Mapa de regimes puros após correção de fronteira'); ax.legend(loc='upper left'); fig.tight_layout(); fig.savefig(results_root/'regime_map_corrected.png',dpi=220); plt.close(fig)

cases=[('base',.25,.25),('constructed',.05,.35),('below_frontier',.5,.65),('on_frontier',.5,.70),('above_frontier',.5,.75)]
mr=[]
for idx,(name,rho,cc) in enumerate(cases):
 p=GameParams(agent_utility=AgentUtilityParams(compromised_reinforced_cost=cc,residual_gain_reinforced=rho))
 eqs=find_mixed_equilibria(p,n_random_starts=16 if name=='constructed' else 6,seed=7+idx,use_global_search=(name=='constructed'),maxiter=1000 if name=='constructed' else 400,snap_tolerance=1e-5,boundary_tolerance=1e-4)
 for n,eq in enumerate(eqs):
  mr.append({'case':name,'rho_R':rho,'c_C':cc,'candidate':n,'classification':eq.classification.value,'x':eq.profile.x,'y':eq.profile.y,'max_regret':eq.report.max_regret,
   'prob_sR_e0':eq.report.information_set_probabilities[(Signal.REINFORCED,Evidence.NO_ALERT)],'prob_sR_e1':eq.report.information_set_probabilities[(Signal.REINFORCED,Evidence.ALERT)],
   'mu_sR_e0':eq.report.posteriors[(Signal.REINFORCED,Evidence.NO_ALERT)],'mu_sR_e1':eq.report.posteriors[(Signal.REINFORCED,Evidence.ALERT)]})
mdf=pd.DataFrame(mr); mdf.to_csv(results_root/'mixed_equilibrium_audit.csv',index=False)
constructed=mdf[mdf.case=='constructed']; ph=constructed[(constructed.x.abs()<1e-5)&(constructed.y.abs()<1e-5)]
cs={'candidates':len(constructed),'boundary_pooling_candidates_near_00':len(ph),'classifications':constructed.classification.value_counts().to_dict()}

profiles={'weak':(.25,.55),'moderate':(.10,.75),'strong':(.03,.95)}; pvals=np.linspace(.001,.999,999); sigma={Signal.BASIC:1.0,Signal.REINFORCED:0.0}; dr=[]
for name,(a,b) in profiles.items():
 det=DetectorParams(alpha_basic=a,beta_basic=b,alpha_reinforced=a,beta_reinforced=b)
 for prior in pvals:
  p=GameParams(prior_compromised=float(prior),detector=det,verifier_utility=base.verifier_utility,agent_utility=base.agent_utility)
  mu0=posterior_compromised(Signal.BASIC,Evidence.NO_ALERT,sigma,sigma,p); mu1=posterior_compromised(Signal.BASIC,Evidence.ALERT,sigma,sigma,p)
  a0='/'.join(x.value for x in best_responses(mu0,p)); a1='/'.join(x.value for x in best_responses(mu1,p)); dr.append({'profile':name,'p':prior,'changes_action':a0!=a1})
dsum=pd.DataFrame(dr).groupby('profile').agg(fraction_action_changed=('changes_action','mean')).reset_index(); dsum.to_csv(results_root/'detector_prior_summary_corrected.csv',index=False)
oldsum=Path('/mnt/data/detector_prior_results/profile_summary.csv'); dcomp={'old_file_available':oldsum.exists()}
if oldsum.exists():
 o=pd.read_csv(oldsum); cmp=dsum.merge(o,on='profile',suffixes=('_new','_old')); cmp['diff']=(cmp.fraction_action_changed_new-cmp.fraction_action_changed_old).abs(); dcomp['max_absolute_difference']=float(cmp['diff'].max())

test=subprocess.run([sys.executable,'-m','pytest','-q'],cwd=root,text=True,capture_output=True,timeout=300)
if test.returncode!=0: raise RuntimeError(test.stdout+'\n'+test.stderr)
report=f'''# Validação da correção de duplicatas de fronteira\n\n## Integração\n\n- `snap_tolerance = 1e-5`: normalização numérica.\n- `boundary_tolerance = 1e-4`: classificação econômica.\n\n## Testes\n\n`{test.stdout.strip()}`\n\n## Caso construído\n\n{json.dumps(cs,ensure_ascii=False,indent=2)}\n\nHá exatamente um candidato próximo de `(x,y)=(0,0)`.\n\n## Mapa rho_R × c_C\n\n- Configurações: {len(df)}\n- Regimes: {json.dumps(counts,ensure_ascii=False)}\n- Concordância analítica: {agreement:.4f}\n- Comparação com mapa anterior: {json.dumps(comparison,ensure_ascii=False)}\n\n## Detector × prior\n\n{dsum.to_string(index=False)}\n\nComparação: {json.dumps(dcomp,ensure_ascii=False)}\n'''
(results_root/'validation_report.md').write_text(report,encoding='utf-8')
print(test.stdout); print('COUNTS',counts); print('COMPARISON',comparison); print('CONSTRUCTED',cs); print('DETECTOR',dcomp)
