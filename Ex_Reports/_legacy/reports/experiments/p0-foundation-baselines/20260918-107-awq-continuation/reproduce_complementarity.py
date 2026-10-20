"""Negative routing check from measured paired outcomes; no simulated router."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
here=Path(__file__).resolve().parent
d=json.loads((here/'data/language_g128_g64_complementarity.json').read_text());s=d['summary']
values=[s['a_successes'],s['b_successes'],s['outcome_oracle_successes']]
fig,ax=plt.subplots(figsize=(7,4));ax.bar(range(3),values,color=['#8cadd2','#ef9e51','#aaaaaa'])
ax.set_xticks(range(3));ax.set_xticklabels(['G128 no clip','G64 no clip','Outcome oracle\nNOT a trained router'])
for i,v in enumerate(values):ax.text(i,v+.8,str(v)+'/'+str(s['episodes']),ha='center')
ax.set_ylim(0,s['episodes']);ax.set_ylabel('Successful paired episodes')
ax.set_title('Existing profiles have no complementary successes');fig.tight_layout()
for ext in ('png','svg'):fig.savefig(here/'figures'/('06_profile_complementarity.'+ext),dpi=160)
plt.close(fig)
