import os
from shutil import copyfile

src_dir = "corpus"
dst_dir = "input"

if __name__ == "__main__":

	copyfile(os.path.join(src_dir, "train_error"), os.path.join(dst_dir, "train.source"))
	copyfile(os.path.join(src_dir, "train_correct"), os.path.join(dst_dir, "train.target"))
	copyfile(os.path.join(src_dir, "validation_full_error"), os.path.join(dst_dir, "validation.source"))
	copyfile(os.path.join(src_dir, "validation_full_correct"), os.path.join(dst_dir, "validation.target"))
	copyfile(os.path.join(src_dir, "test_full_error"), os.path.join(dst_dir, "test.source"))
	copyfile(os.path.join(src_dir, "test_full_correct"), os.path.join(dst_dir, "test.target"))

	k = 0

	while (os.path.isfile(os.path.join(src_dir, "test_" + str(k) + "_error"))):

		source_file = os.path.join(src_dir, "test_" + str(k) + "_error")
		target_file = os.path.join(src_dir, "test_" + str(k) + "_correct")

		test_location = os.path.join(dst_dir, str(k + 1))

		if not os.path.isdir(test_location):

			os.mkdir(test_location)

		copyfile(source_file, os.path.join(test_location, "test.source"))
		copyfile(target_file, os.path.join(test_location, "test.target"))

		k += 1
