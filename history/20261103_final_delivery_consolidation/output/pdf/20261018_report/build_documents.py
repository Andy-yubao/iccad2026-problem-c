from pathlib import Path
import re,json,html,sys
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.colors import HexColor
from reportlab.lib.pagesizes import A4,landscape
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Image,PageBreak,KeepTogether
from PIL import Image as PILImage
OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[2]
pdfmetrics.registerFont(TTFont('YaHei','C:/Windows/Fonts/msyh.ttc',subfontIndex=0))
pdfmetrics.registerFont(TTFont('YaHeiBold','C:/Windows/Fonts/msyhbd.ttc',subfontIndex=0))
pdfmetrics.registerFontFamily('YaHei',normal='YaHei',bold='YaHeiBold',italic='YaHei',boldItalic='YaHeiBold')
styles={
 'body':ParagraphStyle('body',fontName='YaHei',fontSize=10,leading=16.3,spaceAfter=7,wordWrap='CJK',textColor=HexColor('#233142')),
 'h1':ParagraphStyle('h1',fontName='YaHeiBold',fontSize=18,leading=25,spaceAfter=13,textColor=HexColor('#183a52'),wordWrap='CJK'),
 'h2':ParagraphStyle('h2',fontName='YaHeiBold',fontSize=12,leading=18,spaceBefore=10,spaceAfter=7,keepWithNext=True,wordWrap='CJK',textColor=HexColor('#245b78')),
 'caption':ParagraphStyle('caption',fontName='YaHei',fontSize=8.2,leading=12,spaceAfter=9,textColor=HexColor('#566879'),wordWrap='CJK'),
}
def inline(s):
    s=html.escape(s)
    s=re.sub(r'\*\*(.+?)\*\*',r'<b>\1</b>',s)
    return s
def footer(c,d):
    w,h=d.pagesize
    c.setStrokeColor(HexColor('#d4dfe6'));c.line(18*mm,15*mm,w-18*mm,15*mm)
    c.setFont('YaHei',8);c.setFillColor(HexColor('#65798b'))
    c.drawString(18*mm,10*mm,'FloorSet / ICCAD 2026 Problem C  |  2026-10-18')
    c.drawRightString(w-18*mm,10*mm,str(d.page))
def build(md,name,wide=False):
    size=landscape(A4) if wide else A4
    if wide:
        styles['body'].fontSize=9;styles['body'].leading=14
        styles['body'].spaceAfter=6
    width=size[0]-36*mm
    story=[]; para=[]
    def flush():
        if para:story.append(Paragraph(inline(' '.join(para)),styles['body']));para.clear()
    for line in md.splitlines():
        line=line.strip()
        if line=='<!-- pagebreak -->':flush();story.append(PageBreak())
        elif line.startswith('!['):
            flush();m=re.match(r'!\[(.*?)\]\((.*?)\)',line); path=OUT/m[2]
            iw,ih=PILImage.open(path).size; maxh=120*mm if wide else 125*mm
            scale=min(width/iw,maxh/ih)
            story.extend([Image(str(path),width=iw*scale,height=ih*scale),Spacer(1,4*mm)])
        elif line.startswith('# '):flush();story.append(Paragraph(inline(line[2:]),styles['h1']))
        elif line.startswith('## '):flush();story.append(Paragraph(inline(line[3:]),styles['h2']))
        elif line.startswith('> '):flush();story.append(Paragraph(inline(line[2:]),styles['caption']))
        elif line=='---':flush();story.append(Spacer(1,4*mm))
        elif not line:flush()
        elif re.match(r'^(\d+\.|-) ',line):flush();story.append(Paragraph(inline(line),styles['body']))
        else:para.append(line)
    flush()
    doc=SimpleDocTemplate(str(OUT/name),pagesize=size,rightMargin=18*mm,leftMargin=18*mm,topMargin=17*mm,bottomMargin=22*mm,title=name[:-4],author='Yu Chengxiao')
    doc.build(story,onFirstPage=footer,onLaterPages=footer)

original=(ROOT/'docs/report_zhu_20261018.md').read_text(encoding='utf-8')
# Preserve the current report text. Insert figures, explanatory text and breaks.
report=original.replace('## 二、',
 '![官方 Beta 榜质量分](assets/leaderboard.png)\n\n> 图1：按官方综合排名展示质量分；红带为我们的两组开发面板算术均值。数据集及汇总口径不同，仅作参照。\n\n<!-- pagebreak -->\n\n## 二、',1)
report=report.replace('## 三、',
 '> 补充说明：MIB 要求同组模块宽、高相同，因此面积也必须相同。但目标面积为 72、240、56 的三个模块，其允许面积区间分别为 [71.28, 72.72]、[237.60, 242.40]、[55.44, 56.56]，彼此没有交集。也就是说，只要遵守 ±1% 的硬面积约束，就无法同时满足该组的同形要求；考虑官方四位小数的同形判定后，这个例子的冲突仍然成立。\n\n<!-- pagebreak -->\n\n## 三、',1)
report+='\n\n## 布局成果示例\n\n![布局对照](assets/layout_progress.png)\n\n> 图2：官方参考、模型提议与当前最终布局，复用 visualization/ 统一样式。选取固定开发100例按最终 Q 排序第50例（worker_40/layouts_4368.th:89），仅作直观展示，不代表全部实例。颜色表示分组，纹理表示 MIB，L/R/T/B 表示指定边界，红轮廓表示重叠；各栏虚线为各自布局外接框。\n'
(OUT/'report_zhu_20261018_illustrated.md').write_text(report,encoding='utf-8')
build(report,'report_zhu_20261018_illustrated.pdf')
if '--report-only' in sys.argv:
    print('Updated report PDF and Markdown.')
    raise SystemExit(0)

letter='''# Dataset feedback: near-boundary reference labels and incompatible MIB constraints

Subject: FloorSet / ICCAD 2026 Problem C - reproducible dataset issues and request for clarification

Dear Problem C organizers,

I am studying the FloorSet dataset and would like to report two issues that substantially affect the usefulness of the reference layouts as supervision and the interpretation of solution quality. I have attached concrete layout visualizations and reproducible case identifiers.

## 1. Reference blocks often stop just short of their required boundary

In a fixed local development panel of 100 layouts drawn from the released training data, 91 reference layouts violate at least one boundary requirement, with 210 violating blocks in total. For 97 of these blocks, across 50 layouts, all required-edge gaps are at most 1% of the corresponding bounding-box width or height. This is a descriptive threshold; the evaluator's touching tolerance remains 1e-6.

For example, in worker_8/layouts_1232.th, sample 102 (zero-based), block 74 has a top-right boundary requirement (code 6). Its reference XYWH is (85, 319, 23, 24), while the bounding-box top is 344. Moving only this block upward by 1 unit, to (85, 320, 23, 24), satisfies the corner requirement without overlap or any change to dimensions or other blocks. Two further one-unit examples are documented in the attachment.

These appear to be easily correctable reference-label defects: the required boundary is only a small distance away, and a direct legal movement is available. Such labels can teach a supervised model to preserve avoidable boundary errors. Could you explain why these near-boundary gaps remain in the released references, and whether a corrected or optionally repaired reference-label release would be possible?

## 2. MIB groups can conflict mathematically with hard area requirements

In worker_0/layouts_1344.th, sample 82, MIB group 1 contains blocks 33, 42 and 63 with target areas 72, 240 and 56. Their permitted area intervals are [71.28, 72.72], [237.60, 242.40] and [55.44, 56.56]. Identical width and height imply identical area, so these hard area requirements conflict with the intended MIB same-shape requirement.

We also accounted for the evaluator's rounding of widths and heights to four decimal places, rather than relying only on exact-area reasoning. For this example the conservatively relaxed intervals remain disjoint. On the same 100-layout panel, our existing solutions incur 393 MIB violations; a conservative certificate establishes at least 374 unavoidable violations under the published evaluator. The resulting arithmetic-mean neutral-runtime quality lower bound is 1.180663, even when all other losses are optimistically ignored. This is a local-panel bound, not a claim about the hidden set.

We understand from Q&A A4 that reference solutions may violate soft constraints, and from A5/A6 that hard constraints take priority. However, a soft target that cannot be satisfied even in principle is different from a suboptimal reference. Here, at least 95.2% of the observed MIB violations are already covered by the unavoidable lower bound. Could you confirm whether these incompatible MIB groups are intentional, explain their origin, and consider providing compatibility flags or a constraint-consistent dataset variant?

## Clarification and possible collaboration

Q&A A23/A24 state that the released and hidden datasets use the same generation method. Does the hidden set contain the same types of near-boundary labels and MIB/area conflicts, or were any additional repairs or consistency checks applied?

For research validation, would it be possible to obtain access to the hidden benchmark if its release is permitted? If it cannot be shared, could you instead run our frozen solver once on your side and provide quality-only and runtime results, ideally with boundary, grouping and MIB violation counts?

I would also be glad to contribute a reproducible audit and reference-repair checks to help improve the dataset. The attached evidence uses saved development records only; it does not require access to hidden test data.

Thank you for your time and for making FloorSet available.

Best regards,
Chengxiao Yu

Attachment: dataset_issues_evidence_20261018.pdf
'''
letter=letter.replace('## 2. MIB','<!-- pagebreak -->\n\n## 2. MIB')
(OUT/'letter_organizers_20261018.md').write_text(letter,encoding='utf-8')
build(letter,'letter_organizers_20261018.pdf')

ev=json.loads((OUT/'evidence.json').read_text(encoding='utf-8'))
examples=ev['boundary']['examples']
attachment='''# Dataset evidence | Boundary labels

Fixed local development panel: 100 layouts from the released training pool. All indices below are zero-based; coordinates are XYWH.

- Reference boundary violations: 210 blocks across 91 layouts.
- Near-boundary subset: 97 blocks across 50 layouts, using gaps <=1% of the respective bounding-box side. The official touching tolerance remains 1e-6.
- The following translations were checked individually for nonnegative coordinates and zero pairwise overlap (tolerance 1e-6). They preserve every block's dimensions and all other positions, including preplaced blocks. Whole-solution Q and grouping are not claimed to improve.

'''
for j,c in enumerate(examples,1):
    attachment+=f"{j}. {c['case_id']}; block {c['block']}; boundary code {c['code']}. Before {c['before']}; after {c['after']}. Required-edge gaps: {c['gaps']}. Maximum normalized gap: {c['relative_gap']*100:.3f}%.\n\n"
attachment+='''> The translations are separate counterfactuals, not one combined repair. The examples were selected to show small gaps; the aggregate counts above describe the fixed panel, not the whole million-layout dataset.

<!-- pagebreak -->

## Boundary example 1 | Close-up of the one-unit gap

![Boundary close-up](assets/boundary_detail_en.png)

> worker_8/layouts_1232.th:102, block 74. Dashed line: layout top (y=344). The block's top is initially y=343. Translating y=319 to y=320 closes the 1-unit gap; no block changes size, and all other blocks retain their positions. The view is cropped to make the small gap visible; the full layout follows.

<!-- pagebreak -->

## Boundary example 1 | Reference / one-block translation / our saved output

![Boundary example](assets/boundary_1.png)

> worker_8/layouts_1232.th:102, block 74. Right edge already touches; top gap is exactly 1. Middle panel moves y from 319 to 320. All sizes and other positions are unchanged. Unknown metrics show n/a. Red/green boundary badges indicate failed/satisfied requirements.

<!-- pagebreak -->

## MIB incompatibility | Explicit certificate

![MIB permitted area intervals](assets/mib_intervals_en.png)

- Case: worker_0/layouts_1344.th:82. MIB group 1, blocks 33 / 42 / 63.
- Original area intervals: [71.28, 72.72] / [237.60, 242.40] / [55.44, 56.56].
- Conservatively relaxed intervals for round-4 shape buckets: [68.800827, 75.199173] / [235.120827, 244.879173] / [52.960827, 59.039173]. These are still pairwise disjoint: at least 3 shape buckets, hence at least 2 MIB violations.
- The rounding relaxation uses public fixed/preplaced spans to bound rectangle dimensions whenever Q is below the evaluator cap. Capped/infeasible cases are already above the claimed lower bound. This avoids assuming that disjoint exact-area intervals alone prove different rounded shapes at arbitrary aspect ratios.
- Fixed dev100: observed MIB violations 393; certified lower bound 374; arithmetic-mean neutral-runtime Q lower bound 1.180663. Bounds ignore other losses and are not attained-solution claims.

<!-- pagebreak -->

## MIB example | Reference / model proposal / our saved output

![MIB layout](assets/mib_layout.png)

> worker_0/layouts_1344.th:82. MIB group 1 is marked M1 with matching hatching. The reference already contains three distinct shapes in this group. Layouts use the shared visualization/ renderer; metrics come from saved official-evaluator records.

<!-- pagebreak -->

## Reproduction and source record

These figures read existing development JSON records only. No new training, inference, optimization, or sealed-test access was performed. Boundary counterfactuals translate only the identified free block. They demonstrate avoidable boundary violations, without asserting an overall Q improvement or unchanged grouping/connectivity quality.

The published official Q&A (2026-08-27) acknowledges soft violations in reference solutions (A4), gives hard/preplaced priority and the symmetric 1% area rule (A5/A6), and states that hidden cases use the same generation method (A23/A24).

Repository: https://github.com/IntelLabs/FloorSet

Case identifiers are relative to the released FloorSet-Lite training dataset. The development panel is a fixed local selection, not the official public validation set or hidden set. Reported panel statistics must not be generalized to all released data without a broader audit.

The evidence bundle includes layout_evidence.json (reference coordinates, public constraints, area targets, the three one-block translations, and the example MIB certificate), evidence.json (local source paths and SHA-256 hashes), and original-size PNGs. Coordinates alone do not reproduce HPWL or whole-solution Q; those metrics come from the archived evaluator records. Full local sources are listed below.

'''
for s in ev['sources']:
    attachment+=f"> {s['path']}\n\n> SHA-256: {s['sha256']}\n\n"
(OUT/'dataset_issues_evidence_20261018.md').write_text(attachment,encoding='utf-8')
build(attachment,'dataset_issues_evidence_20261018.pdf',wide=True)
print('Created all 3 PDFs and their Markdown sources.')
