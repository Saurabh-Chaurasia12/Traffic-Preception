{/* <script>
  function checkStreamEnd() {
      fetch('/video_feed')
      .then(response => response.text())
      .then(data => {
          if (data.includes("END_STREAM")) {
              console.log("✅ Video has ended. Refreshing in 5 seconds...");
              setTimeout(() => {
                  location.reload();
              }, 5000);
          }
      });
  }

  // Check for stream end every second
  setInterval(checkStreamEnd, 1000);
</script> */}
