import { useState } from "react";
import { Form, Input, Button, Typography, message } from "antd";
import { UserOutlined, LockOutlined } from "@ant-design/icons";
import axios from "axios";

const { Title, Text } = Typography;

const Login = ({ onLogin, onSwitchToSignup }) => {
  const [loading, setLoading] = useState(false);

  const onFinish = async (values) => {
    setLoading(true);
    try {
      const formData = new URLSearchParams();
      formData.append("username", values.username);
      formData.append("password", values.password);
      
      const response = await axios.post("http://127.0.0.1:8000/api/auth/login", formData, {
        headers: { "Content-Type": "application/x-www-form-urlencoded" },
      });
      
      localStorage.setItem("token", response.data.access_token);
      localStorage.setItem("username", values.username);
      message.success("Login successful!");
      onLogin(response.data.access_token);
    } catch (error) {
      message.error(error.response?.data?.detail || "Login failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="dynamic-card" style={{ maxWidth: 400, margin: "4rem auto", padding: "2rem" }}>
      <Title level={2} className="gradient-text" style={{ textAlign: "center", marginBottom: "2rem" }}>
        A smart system for Personalized Interior Design and ARTry-On
      </Title>
      <Title level={3} style={{ textAlign: "center", marginBottom: "2rem", color: "#6b7280", fontWeight: 400 }}>
        Login
      </Title>
      <Form name="login" onFinish={onFinish} layout="vertical" size="large">
        <Form.Item name="username" rules={[{ required: true, message: "Please input your username!" }]}>
          <Input prefix={<UserOutlined />} placeholder="Username" />
        </Form.Item>
        <Form.Item name="password" rules={[{ required: true, message: "Please input your password!" }]}>
          <Input.Password prefix={<LockOutlined />} placeholder="Password" />
        </Form.Item>
        <Form.Item>
          <Button type="primary" htmlType="submit" className="btn-enhanced" loading={loading} block>
            Login
          </Button>
        </Form.Item>
      </Form>
      <Text style={{ textAlign: "center", display: "block", marginTop: "1rem", color: "#6b7280" }}>
        Don't have an account?{" "}
        <Button type="link" onClick={onSwitchToSignup} style={{ padding: 0, color: "#a78bfa" }}>
          Sign up
        </Button>
      </Text>
    </div>
  );
};

export default Login;

