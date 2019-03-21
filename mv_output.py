import os
from shutil import copyfile

src_dir = "output/tmp"
dst_dir = "raw_output"

if __name__ == "__main__":

	k = 1

	while (os.path.isdir(os.path.join(src_dir, str(k)))):

		current_dir = os.path.join(src_dir, str(k))

		source_file = os.path.join(current_dir, "gen.out.org")
		target_file = os.path.join(current_dir, "gen.out.ref")
		output_file = os.path.join(current_dir, "gen.out.sys")

		out_src = os.path.join(dst_dir, "src")
		out_ref = os.path.join(dst_dir, "ref")
		out_sys = os.path.join(dst_dir, "sys")

		if not os.path.isdir(out_src):

			os.mkdir(out_src)

		if not os.path.isdir(out_ref):

			os.mkdir(out_ref)

		if not os.path.isdir(out_sys):

			os.mkdir(out_sys)

		copyfile(source_file, os.path.join(out_src, str(k) + ".out"))
		copyfile(target_file, os.path.join(out_ref, str(k) + ".out"))
		copyfile(output_file, os.path.join(out_sys, str(k) + ".out"))

		k += 1
