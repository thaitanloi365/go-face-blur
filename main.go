package main

import (
	"os"
	"thaitanloi365/go-face-blur/facebluring"
)

func main() {
	var faceDetector = facebluring.New(&facebluring.Config{
		CascadeFile: "./cascade/facefinder",
	})

	out, err := os.OpenFile("./out.jpg", os.O_CREATE|os.O_WRONLY, 0755)
	if err != nil {
		panic(err)
	}
	defer out.Close()

	err = faceDetector.BlurFaces("./3.jpg", out)
	if err != nil {
		panic(err)
	}
}
