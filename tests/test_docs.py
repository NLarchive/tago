"""Documentation numbers must be regenerated from artifacts, never typed by hand."""
import re
from pathlib import Path
from tago.workflows.analysis import run_summary, markdown_report

ROOT = Path(__file__).resolve().parents[1]

def test_readme_results_table_matches_artifact():
    readme = (ROOT / 'README.md').read_text(encoding='utf-8')
    block = re.search(r'<!-- BEGIN GENERATED: v37-results -->\n(.*?)\n<!-- END GENERATED: v37-results -->', readme, re.S).group(1)
    assert block == markdown_report([run_summary(ROOT / 'results' / 'v37_colab_t4_qwen2.5-7b')])

def test_readme_v381_table_matches_artifact():
    from tago.workflows.analysis import markdown_synthetic
    readme = (ROOT / 'README.md').read_text(encoding='utf-8')
    block = re.search(r'<!-- BEGIN GENERATED: v381-results -->\n(.*?)\n<!-- END GENERATED: v381-results -->', readme, re.S).group(1)
    run = ROOT / 'results' / 'v38_1_colab_t4_qwen2.5-7b'
    assert block == markdown_report([run_summary(run)]) + '\n\nSynthetic sweep of the same run:\n\n' + markdown_synthetic(run)

def test_readme_v382_table_matches_artifact():
    from tago.workflows.analysis import markdown_synthetic
    readme = (ROOT / 'README.md').read_text(encoding='utf-8')
    block = re.search(r'<!-- BEGIN GENERATED: v382-results -->\n(.*?)\n<!-- END GENERATED: v382-results -->', readme, re.S).group(1)
    run = ROOT / 'results' / 'v38_2_colab_t4_qwen2.5-7b'
    assert block == markdown_report([run_summary(run)]) + '\n\nSynthetic sweep of the same run:\n\n' + markdown_synthetic(run)

def test_readme_v383_table_matches_artifact():
    from tago.workflows.analysis import markdown_synthetic
    readme = (ROOT / 'README.md').read_text(encoding='utf-8')
    block = re.search(r'<!-- BEGIN GENERATED: v383-results -->\n(.*?)\n<!-- END GENERATED: v383-results -->', readme, re.S).group(1)
    run = ROOT / 'results' / 'v38_3_colab_t4_qwen2.5-7b'
    assert block == markdown_report([run_summary(run)]) + '\n\nSynthetic sweep of the same run:\n\n' + markdown_synthetic(run)

def test_readme_v384_table_matches_artifact():
    from tago.workflows.analysis import markdown_synthetic
    readme = (ROOT / 'README.md').read_text(encoding='utf-8')
    block = re.search(r'<!-- BEGIN GENERATED: v384-results -->\n(.*?)\n<!-- END GENERATED: v384-results -->', readme, re.S).group(1)
    run = ROOT / 'results' / 'v38_4_colab_t4_qwen2.5-7b'
    assert block == markdown_report([run_summary(run)]) + '\n\nSynthetic sweep of the same run:\n\n' + markdown_synthetic(run)

def test_readme_campaign_table_matches_artifact():
    readme = (ROOT / 'README.md').read_text(encoding='utf-8')
    block = re.search(r'<!-- BEGIN GENERATED: v384-campaign -->\n(.*?)\n<!-- END GENERATED: v384-campaign -->', readme, re.S).group(1)
    assert block == (ROOT / 'results' / 'v38_4_campaign_colab_t4' / 'campaign_report.md').read_text(encoding='utf-8').rstrip()

def test_readme_links_exist():
    readme = (ROOT / 'README.md').read_text(encoding='utf-8')
    for link in re.findall(r'\]\(((?!http)[^)#]+)\)', readme):
        assert (ROOT / link).exists(), link

def test_packaging_metadata():
    text = (ROOT / 'pyproject.toml').read_text(encoding='utf-8')
    assert 'name = "tago"' in text and 'requires-python = ">=3.10"' in text
    assert (ROOT / 'LICENSE').exists() and (ROOT / '.github' / 'workflows' / 'ci.yml').exists()

def test_results_report_is_in_sync_with_the_artifacts():
    import subprocess, sys
    assert subprocess.run([sys.executable, 'tools/build_report.py', '--check'], cwd=ROOT).returncode == 0
