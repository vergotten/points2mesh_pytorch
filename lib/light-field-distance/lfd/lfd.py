import argparse
import sys
import os
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import Optional

import numpy as np
import trimesh

SIMILARITY_TAG = b"SIMILARITY:"
CURRENT_DIR = Path(__file__).parent

NAME_TEMPLATES = [
    "{}_q4_v1.8.art",
    "{}_q8_v1.8.art",
    "{}_q8_v1.8.cir",
    "{}_q8_v1.8.ecc",
    "{}_q8_v1.8.fd",
]


def find_similarity_in_logs(logs: bytes) -> float:
    """Get line from the logs where similarity is mentioned.

    Args:
        logs: Unprocessed logs from the docker container after a command was
            run.

    Returns:
        Similarity measure from the log.
    """
    logs = logs.split()
    similarity_line: Optional[bytes] = None
    for index, line in enumerate(logs):
        if line.startswith(SIMILARITY_TAG):
            similarity_line = logs[index + 1]
            break
    return float(similarity_line)

def exist_and_nonempty(path):
    return os.path.exists(path) and os.path.getsize(path) > 0

class MeshEncoder:
    """Class holding an object and preprocessing it using an external cmd."""

    # kiui: modified to save permanent cache
    def __init__(self, mesh: trimesh.Trimesh, cache_dir: Optional[str]=None, name: Optional[str]=None):
        """Instantiate the class.

        It instantiates an empty, temporary folder that will hold any
        intermediate data necessary to calculate Light Field Distance.

        Args:
            vertices: np.ndarray of vertices consisting of 3 coordinates each.
            triangles: np.ndarray where each entry is a vector with 3 elements.
                Each element correspond to vertices that create a triangle.
        """
        self.mesh = mesh
        if cache_dir and name:
            self.cache_dir = Path(cache_dir).resolve() # to absolute path! important.
            os.makedirs(self.cache_dir, exist_ok=True)
            self.file_name = name
            self.permanent = True
        else:
            self.cache_dir = Path(tempfile.mkdtemp())
            self.file_name = uuid.uuid4()
            self.permanent = False

        self.temp_path = self.cache_dir / "{}.obj".format(self.file_name)

        self.mesh.export(self.temp_path.as_posix())

    def get_path(self) -> str:
        """Get path of the object.

        Commands require that an object is represented without any extension.

        Returns:
            Path to the temporary object created in the file system that
            holds the Wavefront OBJ data of the object.
        """
        return self.temp_path.with_suffix("").as_posix()

    def align_mesh(self, verbose=False):
        """Create data of a 3D mesh to calculate Light Field Distance.

        It runs an external command that create intermediate files and moves
        these files to created temporary folder.

        Returns:
            None
        """

        # if already aligned, do nothing.
        if all([exist_and_nonempty((self.cache_dir / f.format(self.file_name)).as_posix()) for f in NAME_TEMPLATES]):
            if verbose:
                print(f'[MeshEncoder.align_mesh] skipped cached alignment for {self.temp_path}')
            return
        else:
            if verbose:
                print(f'[MeshEncoder.align_mesh] generating alignment for {self.temp_path}')

            process = subprocess.Popen(
                ["./3DAlignment", self.get_path()],
                cwd=(CURRENT_DIR / "Executable").as_posix(),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            output, err = process.communicate()
            if len(err) > 0:
                print(err)
                sys.exit(1)

    def __del__(self):
        if not self.permanent:
            shutil.rmtree(self.cache_dir.as_posix())


class LightFieldDistance:
    """Class that allows to calculate light field distance.

    It supports representing objects in the Wavefront OBJ format.
    """

    def __init__(self, verbose: bool = 0):
        """Instantiate the class.

        Args:
            verbose: Whether to display processing information performed step
                by step.
        """
        self.verbose = verbose

    def get_distance(
        self,
        mesh_1: trimesh.Trimesh,
        mesh_2: trimesh.Trimesh,
    ) -> float:
        """Calculate LFD between two meshes.

        These objects are taken as meshes from the Wavefront OBJ format. Hence
        vertices represent coordinates as a matrix Nx3, while `triangles`
        connects these vertices. Each entry in the `triangles` is a 3 element
        vector consisting of indices to appropriate vertices.

        Args:
            

        Returns:
            Light Field Distance between `object_1` and `object_2`.
        """
        mesh_1 = MeshEncoder(mesh_1)
        mesh_2 = MeshEncoder(mesh_2)
        
        mesh_1.align_mesh(self.verbose)
        mesh_2.align_mesh(self.verbose)

        if self.verbose:
            print("Calculating distances ...")

        process = subprocess.Popen(
            ["./Distance", mesh_1.get_path(), mesh_2.get_path()],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=(CURRENT_DIR / "Executable").as_posix(),
        )

        output, err = process.communicate()
        lfd = find_similarity_in_logs(output)

        return lfd

# support permanant cache:
def light_field_distance(mesh_1: MeshEncoder, mesh_2: MeshEncoder):
    process = subprocess.Popen(
        ["./Distance", mesh_1.get_path(), mesh_2.get_path()],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=(CURRENT_DIR / "Executable").as_posix(),
    )
    output, err = process.communicate()
    lfd = find_similarity_in_logs(output)
    return lfd


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Script that generates score for two shapes saved in Wavefront "
            "OBJ format"
        )
    )
    parser.add_argument(
        "file1",
        type=str,
        help="Path to the first *.obj file in Wavefront OBJ format",
    )

    parser.add_argument(
        "file2",
        type=str,
        help="Path to the second *obj file in Wavefront OBJ format",
    )

    args = parser.parse_args()

    lfd_calc = LightFieldDistance(verbose=True)

    mesh_1: trimesh.Trimesh = trimesh.load(args.file1)
    mesh_2: trimesh.Trimesh = trimesh.load(args.file2)

    lfd = lfd_calc.get_distance(
        mesh_1.vertices, mesh_1.faces, mesh_2.vertices, mesh_2.faces
    )
    print("LFD: {:.4f}".format(lfd))


if __name__ == "__main__":
    main()
