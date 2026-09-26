"""A paper rebuild must not discard subsequent proof statuses or review notes."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'paper'))
from ledger_sections import replace_section


def test_rebuilding_older_sections_preserves_newer_claims_and_review_notes(tmp_path):
    p = tmp_path/'ledger.md'
    later = '## New proof\n\nM09: external review pending.\n'
    p.write_text('# Ledger\n\nOld intro.\n\n## Old experiment\n\nOld counts.\n\n'+later)
    for _ in range(2):
        replace_section(p, None, '# Ledger\n\nUpdated intro.\n')
        replace_section(p, '## Old experiment', '## Old experiment\n\nCorrected counts.\n')
        assert p.read_text().endswith(later)
        assert p.read_text().count('## Old experiment') == 1
        assert 'Old counts' not in p.read_text()


def test_a_new_section_is_added_without_overwriting_existing_sections(tmp_path):
    p = tmp_path/'ledger.md'
    p.write_text('# Ledger\n\n## Review\n\nKeep this.\n')
    replace_section(p, '## Result', '## Result\n\nNew result.\n')
    assert '## Review\n\nKeep this.' in p.read_text()
    assert p.read_text().endswith('## Result\n\nNew result.\n')
