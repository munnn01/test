"""Plot actual four-point task curves; never smooth an invalid BD-rate curve."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs',nargs=3,type=Path,required=True)
    p.add_argument('--scope',required=True,choices=['DEV','TEST'])
    p.add_argument('--out',type=Path,required=True,help='output filename without extension')
    args=p.parse_args()
    reports={s['task']:s for s in [json.loads((run/'summary.json').read_text(encoding='utf-8')) for run in args.runs]}
    plt.rcParams.update({'font.size':10,'font.family':'DejaVu Sans','pdf.fonttype':42,'svg.fonttype':'none'})
    fig,axes=plt.subplots(3,2,figsize=(9,10),layout='constrained',sharex='row',sharey='row')
    metrics={'ar':'Top-1 accuracy','od':'COCO AP','ot':'TrackEval HOTA / 100'}
    for row,task in enumerate(('ar','od','ot')):
        for column,codec in enumerate(('h264','h265')):
            ax=axes[row,column]; report=reports[task]; curves=report['curves'][codec]
            for arm,color,marker,label in [('anchor','#0072B2','o','Unprocessed anchor'),('adaptive','#D55E00','s','v34r1')]:
                curve=curves[arm]
                ax.plot(curve['rate'],curve['quality'],color=color,marker=marker,label=label,linewidth=1.4,markersize=4)
                if arm=='adaptive':
                    for qp,x,y in zip((30,35,40,45),curve['rate'],curve['quality']):
                        ax.annotate(str(qp),(x,y),xytext=(4,5),textcoords='offset points',fontsize=8,color=color)
            result=report['comparisons'][codec]['adaptive']
            status=f"BD-rate {result['pchip_bd_rate_pct']:.2f}%" if result['valid'] else 'BD-rate undefined: '+result['invalid_reason']
            count=report['grid_audit']['source_count']
            unit={'ar':'videos','od':'images','ot':'windows'}[task]
            ax.set_title(task.upper()+' / '+codec.upper()+f' ({count} {unit})'+'\n'+status,fontsize=10)
            ax.set_xlabel('Packet bits per canonical RGB pixel')
            ax.set_ylabel(metrics[task]); ax.grid(alpha=.2); ax.legend(fontsize=8,loc='best')
            ax.margins(x=.12,y=.17)
    fig.suptitle('v34r1 '+args.scope+' — four actual QPs: 30, 35, 40, 45',fontsize=13)
    fig.supxlabel('Lines join measured points. No curve repair or extrapolation. Finite declared cohorts; no all-task confidence claim.',fontsize=8)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    for suffix in ('png','pdf'): fig.savefig(args.out.with_suffix('.'+suffix),dpi=220)
    plt.close(fig)
    print(args.out.with_suffix('.png'))


if __name__=='__main__': main()
