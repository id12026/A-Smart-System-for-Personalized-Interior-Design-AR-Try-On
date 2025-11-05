const Footer = ({ isDarkMode }) => {
  return (
    <footer
      style={{
        background: "rgba(255, 255, 255, 0.5)",
        backdropFilter: "blur(15px)",
        padding: "1.5rem 1rem",
        textAlign: "center",
        marginTop: "4rem",
        borderTop: "1px solid rgba(200, 150, 255, 0.4)",
        boxShadow: "0 -4px 30px rgba(167, 139, 250, 0.15)",
        color: "#1a1a1a",
      }}
    />
  );
};

export default Footer;
