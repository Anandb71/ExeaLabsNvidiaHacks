"""Exercise the actual Swift ZIP writer with invalid source paths, offline on macOS."""

import platform
import shutil
import subprocess
import zipfile

import pytest
from conftest import REPO

pytestmark = pytest.mark.skipif(
    platform.system() != "Darwin" or shutil.which("swiftc") is None, reason="needs swiftc on macOS"
)


@pytest.fixture(scope="module")
def zip_check(tmp_path_factory):
    folder = tmp_path_factory.mktemp("zip_swift")
    main = folder / "main.swift"
    main.write_text("""import Foundation
 do {
    try ZipWriter.zip(folder: URL(fileURLWithPath: CommandLine.arguments[1]),
                      to: URL(fileURLWithPath: CommandLine.arguments[2]))
 } catch {
    FileHandle.standardError.write(Data("\\(error.localizedDescription)\\n".utf8))
    exit(1)
 }
""")
    exe = folder / "zip_check"
    subprocess.run(
        ["swiftc", str(main), str(REPO / "ios/ArrivalCapture/ArrivalCapture/FrameBundle.swift"), "-o", str(exe)],
        check=True,
        capture_output=True,
        timeout=120,
    )
    return exe


@pytest.mark.parametrize("source_kind", ["missing", "file"])
@pytest.mark.parametrize("existing", [False, True])
def test_invalid_zip_source_preserves_destination(zip_check, tmp_path, source_kind, existing):
    source = tmp_path / "source"
    if source_kind == "file":
        source.write_text("not a folder")
    output = tmp_path / "frames.zip"
    if existing:
        output.write_bytes(b"previous archive")
    result = subprocess.run([str(zip_check), str(source), str(output)], capture_output=True, text=True, timeout=10)
    assert result.returncode == 1
    assert result.stderr.strip()
    if existing:
        assert output.read_bytes() == b"previous archive"
    else:
        assert not output.exists()


def test_empty_directory_is_still_a_valid_zip_source(zip_check, tmp_path):
    source = tmp_path / "empty"
    source.mkdir()
    output = tmp_path / "empty.zip"
    subprocess.run([str(zip_check), str(source), str(output)], check=True, capture_output=True, timeout=10)
    with zipfile.ZipFile(output) as archive:
        assert archive.namelist() == []
        assert archive.testzip() is None
