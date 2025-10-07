package mdp

import (
	"fmt"
	"io"
	"log"
	"net/http"
	"os"

	"github.com/go-echarts/go-echarts/v2/charts"
	"github.com/go-echarts/go-echarts/v2/components"
	"github.com/go-echarts/go-echarts/v2/opts"
)

func Plot(policyResults ...PolicyAverageReward) {
	numSteps := len(policyResults[0].AverageRewards)

	line := charts.NewLine()
	line.SetGlobalOptions(
		charts.WithTitleOpts(opts.Title{
			Title: "multi lines",
		}),
		charts.WithInitializationOpts(opts.Initialization{
			Theme: "shine",
		}),
	)

	var steps []string
	for i := 0; i < numSteps; i++ {
		steps = append(steps, fmt.Sprintf("%d", i))
	}

	line = line.SetXAxis(steps)
	for _, policyResults := range policyResults {
		items := make([]opts.LineData, 0)
		for i := 0; i < numSteps; i++ {
			items = append(items, opts.LineData{Value: policyResults.AverageRewards[i]})
		}

		line.AddSeries(policyResults.Policy.Name(), items)
	}

	page := components.NewPage()
	page.AddCharts(
		line,
	)
	err := os.MkdirAll("charts",0700)
	chk(err)
	f, err := os.Create("charts/testbed.html")
	chk(err)
	err = page.Render(io.MultiWriter(f))
	chk(err)

	fs := http.FileServer(http.Dir("charts"))
	log.Println("runnning server at http://localhost:8089")
	log.Fatal(http.ListenAndServe("localhost: 8089", fs))
}

var chk = func(err error) {
	if err != nil {panic(err)}
}


