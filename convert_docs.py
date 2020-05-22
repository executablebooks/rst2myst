# python3

from pathlib import Path
import jupytext
import os


def convert_to_myst(file, dest_path):
	# Read a notebook from a file
	ntbk = jupytext.read(file, fmt='ipynb')
	# Write notebook to md file
	jupytext.write(ntbk, dest_path.joinpath(file.stem + '.md'), fmt='md')


def main():
	root_path = Path(__file__).parent.resolve()
	book_path = root_path.joinpath('quantecon-book')
	source_path = book_path.joinpath('source')
	docs_path = book_path.joinpath('docs')

	for repo in os.listdir(source_path):
		dir_path = source_path.joinpath(repo)
		dest_path = docs_path.joinpath(repo)
		if not dest_path.is_dir():
			os.mkdir(dest_path)
		for file in os.listdir(dir_path):
			if file.endswith('.ipynb') and 'test' not in file:
				convert_to_myst(dir_path.joinpath(file), dest_path)


if __name__ == '__main__':
	main()
