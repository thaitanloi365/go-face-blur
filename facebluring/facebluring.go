package facebluring

import (
	"errors"
	"fmt"
	"image"
	"image/jpeg"
	"image/png"
	"io"
	"io/ioutil"
	"os"
	"path/filepath"

	"github.com/disintegration/imaging"
	pigo "github.com/esimov/pigo/core"
	"github.com/fogleman/gg"
)

// Config config
type Config struct {
	Angle        float64
	CascadeFile  string
	Destination  string
	MinSize      int
	MaxSize      int
	ShiftFactor  float64
	ScaleFactor  float64
	IouThreshold float64
	Puploc       string
	Flploc       string
	MarkDetEyes  bool
}

// FaceBluring bluring
type FaceBluring struct {
	dc         *gg.Context
	fd         *Config
	plc        *pigo.PuplocCascade
	flpcs      map[string][]*pigo.FlpCascade
	classifier *pigo.Pigo
}

// New init
func New(config *Config) *FaceBluring {
	var instance = &FaceBluring{
		fd: config,
	}

	if instance.fd == nil {
		instance.fd = &Config{}
	}

	if instance.fd.MinSize == 0 {
		instance.fd.MinSize = 20
	}

	if instance.fd.MaxSize == 0 {
		instance.fd.MaxSize = 1000
	}

	if instance.fd.ShiftFactor == 0 {
		instance.fd.ShiftFactor = 0.1
	}

	if instance.fd.ScaleFactor == 0 {
		instance.fd.ScaleFactor = 1.1
	}

	if instance.fd.IouThreshold == 0 {
		instance.fd.IouThreshold = 0.2
	}

	cascadeFile, err := ioutil.ReadFile(instance.fd.CascadeFile)
	if err != nil {
		panic(fmt.Errorf("Can not open cascade file %s error: %v", instance.fd.CascadeFile, err))
	}

	var p = pigo.NewPigo()
	// Unpack the binary file. This will return the number of cascade trees,
	// the tree depth, the threshold and the prediction from tree's leaf nodes.
	classifier, err := p.Unpack(cascadeFile)
	if err != nil {
		panic(fmt.Errorf("Unpack cascade file error: %v", err))
	}

	if len(instance.fd.Puploc) > 0 {
		pl := pigo.NewPuplocCascade()
		cascade, err := ioutil.ReadFile(instance.fd.Puploc)
		if err != nil {
			panic(fmt.Errorf("Can not open Puploc file %s error: %v", instance.fd.Puploc, err))
		}
		plc, err := pl.UnpackCascade(cascade)
		if err != nil {
			panic(fmt.Errorf("Unpack cascade Puploc file error: %v", err))
		}

		if len(instance.fd.Flploc) > 0 {
			flpcs, err := pl.ReadCascadeDir(instance.fd.Flploc)
			if err != nil {
				panic(fmt.Errorf("Read cascade dir error: %v", err))
			}
			instance.flpcs = flpcs
		}
		instance.plc = plc
	}

	instance.classifier = classifier

	return instance
}

func (f *FaceBluring) detectFaces(source string) ([]pigo.Detection, error) {
	srcFile, err := os.Open(source)
	if err != nil {
		return nil, fmt.Errorf("Can not open %s error: %v", source, err)
	}
	defer srcFile.Close()

	src, err := pigo.DecodeImage(srcFile)
	if err != nil {
		return nil, fmt.Errorf("Decode image error: %v", err)
	}

	pixels := pigo.RgbToGrayscale(src)
	cols, rows := src.Bounds().Max.X, src.Bounds().Max.Y

	f.dc = gg.NewContext(cols, rows)
	f.dc.DrawImage(src, 0, 0)

	var imgParams = &pigo.ImageParams{
		Pixels: pixels,
		Rows:   rows,
		Cols:   cols,
		Dim:    cols,
	}

	cParams := pigo.CascadeParams{
		MinSize:     f.fd.MinSize,
		MaxSize:     f.fd.MaxSize,
		ShiftFactor: f.fd.ShiftFactor,
		ScaleFactor: f.fd.ScaleFactor,
		ImageParams: *imgParams,
	}

	// Run the classifier over the obtained leaf nodes and return the detection results.
	// The result contains quadruplets representing the row, column, scale and detection score.
	faces := f.classifier.RunCascade(cParams, f.fd.Angle)

	// Calculate the intersection over union (IoU) of two clusters.
	faces = f.classifier.ClusterDetections(faces, f.fd.IouThreshold)

	return faces, nil
}

func (f *FaceBluring) blurFaces(faces []pigo.Detection) {
	for _, face := range faces {
		var rect = image.Rect(
			face.Col-face.Scale/2,
			face.Row-face.Scale/2,
			face.Col+face.Scale/2,
			face.Row+face.Scale/2,
		)
		var faceZone = imaging.Crop(f.dc.Image(), rect)
		var blurImage = imaging.Blur(faceZone, 5.0)
		f.dc.DrawImage(blurImage, face.Col-face.Scale/2, face.Row-face.Scale/2)
	}
}

func (f *FaceBluring) encodeImage(dst io.Writer) error {
	var err error
	var img = f.dc.Image()

	switch dst.(type) {
	case *os.File:
		ext := filepath.Ext(dst.(*os.File).Name())
		switch ext {
		case "", ".jpg", ".jpeg":
			err = jpeg.Encode(dst, img, &jpeg.Options{Quality: 100})
		case ".png":
			err = png.Encode(dst, img)
		default:
			err = errors.New("unsupported image format")
		}
	default:
		err = jpeg.Encode(dst, img, &jpeg.Options{Quality: 100})
	}
	return err
}

// BlurFaces blur faces
func (f *FaceBluring) BlurFaces(source string, dst io.Writer) error {
	faces, err := f.detectFaces(source)
	if err != nil {
		return err
	}
	f.blurFaces(faces)

	return f.encodeImage(dst)
}
