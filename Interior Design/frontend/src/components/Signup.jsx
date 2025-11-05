import { useState } from "react";
import { Form, Input, Button, Typography, message } from "antd";
import { UserOutlined, LockOutlined, MailOutlined } from "@ant-design/icons";
import axios from "axios";

const { Title, Text } = Typography;

const Signup = ({ onSignup, onSwitchToLogin }) => {
  const [loading, setLoading] = useState(false);

  const onFinish = async (values) => {
    if (values.password !== values.confirmPassword) {
      message.error("Passwords do not match!");
      return;
    }
    setLoading(true);
    try {
      const response = await axios.post("http://127.0.0.1:8000/api/auth/signup", {
        email: values.email,
        username: values.username,
        password: values.password,
      });
      
      message.success("Signup successful! Please login.");
      onSwitchToLogin();
    } catch (error) {
      message.error(error.response?.data?.detail || "Signup failed");
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
        Sign Up
      </Title>
      <Form name="signup" onFinish={onFinish} layout="vertical" size="large">
        <Form.Item name="email" rules={[{ required: true, type: "email", message: "Please input a valid email!" }]}>
          <Input prefix={<MailOutlined />} placeholder="Email" />
        </Form.Item>
        <Form.Item name="username" rules={[{ required: true, message: "Please input your username!" }]}>
          <Input prefix={<UserOutlined />} placeholder="Username" />
        </Form.Item>
        <Form.Item name="password" rules={[{ required: true, min: 6, message: "Password must be at least 6 characters!" }]}>
          <Input.Password prefix={<LockOutlined />} placeholder="Password" />
        </Form.Item>
        <Form.Item name="confirmPassword" rules={[{ required: true, message: "Please confirm your password!" }]}>
          <Input.Password prefix={<LockOutlined />} placeholder="Confirm Password" />
        </Form.Item>
        <Form.Item>
          <Button type="primary" htmlType="submit" className="btn-enhanced" loading={loading} block>
            Sign Up
          </Button>
        </Form.Item>
      </Form>
      <Text style={{ textAlign: "center", display: "block", marginTop: "1rem", color: "#6b7280" }}>
        Already have an account?{" "}
        <Button type="link" onClick={onSwitchToLogin} style={{ padding: 0, color: "#a78bfa" }}>
          Login
        </Button>
      </Text>
    </div>
  );
};

export default Signup;

