"""Rebuild searchable support documents and figures from the archived PDFs.

Requires pypdf and pdfplumber. Does not rewrite the manually reviewed problem
statement. Source spelling and contradictions are intentionally retained.
"""
from pathlib import Path
import re
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / 'references/official'


def clean_page(page):
    lines = page.extract_text().splitlines()
    lines = [l.rstrip() for l in lines if l.strip()]
    if lines and re.fullmatch(r'\d+', lines[-1].strip()):
        lines.pop()
    return '\n'.join(lines)


def paragraphs(text):
    text = re.sub(r'(?<=\w)-\n(?=[a-z])', '', text)
    return re.sub(r'\s+', ' ', text).strip()


def main():
    qa_source = DOCS / 'C_QA_20260827.pdf'
    pages = [clean_page(p) for p in PdfReader(qa_source).pages]
    joined = '\n'.join(pages)
    # Match only Q1./A1. style markers, preserving all long questions and answers.
    tokens = re.split(r'\b([QA]\d{1,2})\.\s*', joined)
    output = ['# Problem C — Official Q&A (2026-08-27)',
              '> 官方英文原文的 Markdown 转写。来源：[PDF](C_QA_20260827.pdf)。',
              '> 保留全部 Q1–Q29 / A1–A29；仅整理断行、问答标题。未擅自修正 A18 等可疑回答。',
              paragraphs(tokens[0])]
    for marker, body in zip(tokens[1::2], tokens[2::2]):
        output.extend([f'## {marker}', paragraphs(body)])
    (DOCS / 'official_qa.md').write_text('\n\n'.join(output) + '\n', encoding='utf-8')
    if len(tokens[1::2]) != 58:
        raise ValueError('Expected 29 questions and 29 answers')

    source = DOCS / 'C_Submission_Guidelines_20260616.pdf'
    text = '\n\n'.join(clean_page(p) for p in PdfReader(source).pages)
    (DOCS / 'submission_guidelines.md').write_text(
        '# Problem C — Submission Guidelines\n\n'
        '> 来源：[官方 PDF](C_Submission_Guidelines_20260616.pdf)。正文仅整理换行。\n'
        '> 阅读前注意：CUDA 12.5 已被 [Q&A A22](official_qa.md#a22) 更正为 13.0；'
        '硬件内存规格有多处不同表述，见[研究导读](../../docs/research_guide.zh-CN.md)。\n\n'
        '```text\n' + text + '\n```\n', encoding='utf-8')

    spec = PdfReader(ROOT / 'references/official/C_20260522.pdf')
    # Retain the original 46-entry bibliography in order, joining print hyphenation.
    reftext = '\n'.join(clean_page(p) for p in spec.pages[8:])
    reftext = reftext.split('References', 1)[1]
    reftext = re.sub(r'6Submissions found.*?will be disqualified\.', '', reftext, flags=re.S)
    entries = re.split(r'\[(\d+)\]\s*', reftext)
    references = ['# 赛题参考文献（原文顺序）',
                  '> 来源：[赛题 PDF](C_20260522.pdf)，第 9–13 页。'
                  '仅恢复断行和断词，保留原文出版年份等信息；不是全部文献已下载的清单。']
    for number, body in zip(entries[1::2], entries[2::2]):
        references.append(f'{number}. {paragraphs(body)}')
    if len(entries[1::2]) != 46:
        raise ValueError('Expected 46 bibliography entries')
    (DOCS / 'problem_references.md').write_text('\n\n'.join(references) + '\n', encoding='utf-8')

    import pdfplumber
    figures = DOCS / 'assets'
    figures.mkdir(exist_ok=True)
    with pdfplumber.open(ROOT / 'references/official/C_20260522.pdf') as pdf:
        page = pdf.pages[1]
        # Original Figure 1 occupies this rectangle in PDF points (not pixels).
        page.crop((160, 70, 440, 255)).to_image(resolution=180).save(figures / 'problem-figure-1.png')
    print('Converted Q&A (29 pairs), submission guide, bibliography (46 entries), Figure 1')


if __name__ == '__main__':
    main()
