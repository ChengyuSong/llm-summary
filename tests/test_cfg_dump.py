"""Tests for parse_cfg_dump and format_annotated_source with ko-clang CFG output."""

from pathlib import Path

from llm_summary.bbid_extractor import format_annotated_source, parse_cfg_dump

FIXTURES = Path(__file__).parent / "fixtures"
CFG_FILE = FIXTURES / "test_cfg_dump.txt"
SOURCE_FILE = FIXTURES / "test_cfg_source.c"


def test_parse_cfg_dump_count() -> None:
    infos = parse_cfg_dump(str(CFG_FILE))
    assert len(infos) == 6


def test_parse_cfg_dump_sorted_by_bbid() -> None:
    infos = parse_cfg_dump(str(CFG_FILE))
    bb_ids = [i.bb_id for i in infos]
    assert bb_ids == sorted(bb_ids)


def test_parse_cfg_dump_conditional() -> None:
    infos = parse_cfg_dump(str(CFG_FILE))
    by_id = {i.bb_id: i for i in infos}

    # BB 0: conditional at line 4
    bb0 = by_id[0]
    assert bb0.is_conditional
    assert bb0.true_bb_id == 1
    assert bb0.false_bb_id == 3
    assert bb0.line == 4

    # BB 1: conditional at line 7
    bb1 = by_id[1]
    assert bb1.is_conditional
    assert bb1.true_bb_id == 2
    assert bb1.false_bb_id == 5
    assert bb1.line == 7


def test_parse_cfg_dump_unconditional() -> None:
    infos = parse_cfg_dump(str(CFG_FILE))
    by_id = {i.bb_id: i for i in infos}

    # BB 2: unconditional at line 10
    bb2 = by_id[2]
    assert not bb2.is_conditional
    assert bb2.line == 10

    # BB 3: unconditional at line 16
    bb3 = by_id[3]
    assert not bb3.is_conditional
    assert bb3.line == 16


def test_parse_cfg_dump_return() -> None:
    infos = parse_cfg_dump(str(CFG_FILE))
    by_id = {i.bb_id: i for i in infos}

    # BB 5: return at line 18
    bb5 = by_id[5]
    assert not bb5.is_conditional
    assert bb5.line == 18


def test_parse_cfg_dump_file_path() -> None:
    infos = parse_cfg_dump(str(CFG_FILE))
    for info in infos:
        assert info.file == "/tmp/test_cfg_dump/test.c"


def test_format_annotated_source() -> None:
    infos = parse_cfg_dump(str(CFG_FILE))
    # Remap file paths to match the fixture source file
    source_name = SOURCE_FILE.name
    for info in infos:
        info.file = str(SOURCE_FILE)

    annotated = format_annotated_source(infos, str(SOURCE_FILE))
    lines = annotated.splitlines()

    # Line 4 (if x > 0) should have BB:0 cond annotation
    line4 = lines[3]  # 0-indexed
    assert "BB:0" in line4
    assert "cond" in line4
    assert "T:1" in line4
    assert "F:3" in line4

    # Line 7 (if x > 10) should have BB:1 cond annotation
    line7 = lines[6]
    assert "BB:1" in line7
    assert "T:2" in line7
    assert "F:5" in line7

    # Line 5 (p[0] = x * 2) should have no annotation
    line5 = lines[4]
    assert "BB:" not in line5


def test_parse_cfg_dump_empty_file(tmp_path: Path) -> None:
    empty = tmp_path / "empty.txt"
    empty.write_text("")
    infos = parse_cfg_dump(str(empty))
    assert infos == []


def test_parse_cfg_dump_no_debug_info() -> None:
    """Fallback format: funcname,0,0:bbid:TYPE."""
    import tempfile

    content = "main,0,0:100:D:101\nmain,0,0:101:R\n"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write(content)
        f.flush()
        infos = parse_cfg_dump(f.name)

    assert len(infos) == 2
    assert infos[0].bb_id == 100
    assert infos[0].file == "main"
    assert infos[0].line == 0
    assert infos[1].bb_id == 101
