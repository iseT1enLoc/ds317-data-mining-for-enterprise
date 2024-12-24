import { Box, TextField, Button, Typography, Stack } from "@mui/material";

const Login = () => {
  const handleSubmit = (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const formData = new FormData(event.currentTarget);
    const formJson = Object.fromEntries(formData.entries());
    alert(JSON.stringify(formJson));
  };

  return (
    <Box
      sx={{
        position: "fixed",
        inset: 0,
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        bgcolor: "#f3d1dc",
      }}
    >
      <Box
        component="div"
        sx={{
          bgcolor: "white",
          width: 320,
          p: 4,
          borderRadius: 2,
          boxShadow: 3,
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
        }}
      >
        <Typography
          variant="h4"
          component="p"
          fontWeight="bold"
          color="primary"
          sx={{ mb: 3 }}
        >
          Sign In
        </Typography>
        <Box
          component="form"
          onSubmit={handleSubmit}
          sx={{ width: "100%" }}
          noValidate
        >
          <Stack spacing={2}>
            <TextField
              label="Account"
              placeholder="Enter your account!"
              variant="outlined"
              fullWidth
              required
            />
            <TextField
              label="Password"
              type="password"
              placeholder="Enter your password"
              variant="outlined"
              fullWidth
              required
            />
            <Button
              type="submit"
              variant="contained"
              color="primary"
              size="large"
              fullWidth
              sx={{ mt: 2 }}
            >
              Submit
            </Button>
          </Stack>
        </Box>
      </Box>
    </Box>
  );
};

export default Login;
